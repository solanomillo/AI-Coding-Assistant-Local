"""
Servicio optimizado para análisis de repositorios.
Soporte especial para proyectos Django con detección estricta.
"""

import zipfile
import shutil
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import logging
from datetime import datetime
import re
import hashlib
import os
import time
import stat

from domain.models.repository import Repository, CodeFile, Function, Class
from infrastructure.database.mysql_repository import MySQLRepository
from infrastructure.parsers import PythonParser, JavaScriptParser, HTMLParser, CSSParser
from application.services.cache_service import CacheService

logger = logging.getLogger(__name__)


class RepositoryService:
    """
    Servicio optimizado para análisis de repositorios.
    """
    
    def __init__(self):
        """Inicializa el servicio."""
        self.db = MySQLRepository()
        self.cache = CacheService()
        
        # Registro de parsers por lenguaje
        self.parsers = {
            '.py': PythonParser(),
            '.js': JavaScriptParser(),
            '.jsx': JavaScriptParser(),
            '.ts': JavaScriptParser(),
            '.tsx': JavaScriptParser(),
            '.html': HTMLParser(),
            '.htm': HTMLParser(),
            '.css': CSSParser(),
            '.scss': CSSParser(),
            '.sass': CSSParser()
        }
        
        # Directorios del sistema
        self.repositories_dir = Path("data/repositories")
        self.repositories_dir.mkdir(parents=True, exist_ok=True)
        
        # Directorios a ignorar (Django y otros)
        self.ignore_dirs = {
            'venv', 'env', '.venv', '__pycache__',
            'node_modules', '.git', '.idea', '.vscode',
            'dist', 'build', 'target', 'logs', 'tmp', 'temp',
            '.github', '.gitlab', '.circleci', 'assets', 'images',
            'migrations',           # Archivos de migración de Django
            '.pytest_cache',        # Caché de pytest
            '.mypy_cache',          # Caché de mypy
            'staticfiles',          # Archivos estáticos compilados
            'media',                # Archivos subidos por usuarios
            'locale'                # Archivos de internacionalización
        }
        
        # Archivos a ignorar
        self.ignore_files = {
            '.env', '.gitignore', '.dockerignore', '.eslintignore',
            'package-lock.json', 'yarn.lock', 'poetry.lock',
            'requirements.txt', 'Pipfile', 'pyproject.toml',
            '*.pyc', '*.pyo', '*.so', '*.dll', '*.exe',
            '*.png', '*.jpg', '*.jpeg', '*.gif', '*.ico', '*.svg',
            '*.mp4', '*.mp3', '*.pdf', '*.doc', '*.docx',
            '*.log', '*.tmp', '*.cache', '*.db', '*.sqlite',
            '.DS_Store', 'Thumbs.db',
            'manage.py',            # Archivo de gestión de Django
            'settings.py',          # Configuración de Django
            'local_settings.py',    # Configuración local
            'wsgi.py',              # WSGI de Django
            'asgi.py',              # ASGI de Django
            '*.sql',                # Archivos SQL
            'dump.rdb',             # Redis dump
            'celerybeat-schedule'   # Celery schedule
        }
        
        # Patrones de archivos a ignorar (wildcard)
        self.ignore_patterns = [
            '*.pyc', '*.pyo', '*.so', '*.dll', '*.exe',
            '*.png', '*.jpg', '*.jpeg', '*.gif', '*.ico', '*.svg',
            '*.mp4', '*.mp3', '*.pdf', '*.doc', '*.docx',
            '*.log', '*.tmp', '*.cache', '*.db', '*.sqlite',
            '*_test.py',            # Archivos de test
            'test_*.py',            # Archivos de test
            'migrations/*.py'       # Migraciones de Django
        ]
        
        # Estadísticas
        self.stats = {
            'total_repos': 0,
            'total_files': 0,
            'total_functions': 0,
            'total_classes': 0,
            'languages': {},
            'django_projects': 0
        }
        
        logger.info("RepositoryService inicializado")
        logger.info(f"Lenguajes soportados: {len(self.parsers)}")
        self._log_supported_languages()
    
    def _log_supported_languages(self) -> None:
        """Registra los lenguajes soportados en el log."""
        languages = {}
        for ext, parser in self.parsers.items():
            lang = parser.language
            if lang not in languages:
                languages[lang] = []
            languages[lang].append(ext)
        
        logger.info("Lenguajes soportados:")
        for lang, exts in languages.items():
            logger.info(f"  - {lang}: {', '.join(exts)}")
    
    def _is_django_project(self, directory: Path) -> bool:
        """
        Detecta si el repositorio es un proyecto Django.
        Requiere al menos 2 indicadores fuertes o 1 fuerte + 2 débiles.
        """
        strong_indicators = {
            'manage.py': False,
            'settings.py': False,
            'wsgi.py': False,
            'asgi.py': False
        }
        
        weak_indicators = {
            'urls.py': 0,
            'models.py': 0,
            'views.py': 0,
            'admin.py': 0,
            'apps.py': 0
        }
        
        for root, dirs, files in os.walk(directory):
            if any(ignore in root for ignore in ['venv', 'env', '.venv', '__pycache__', 'migrations']):
                continue
            
            for file in files:
                if file in strong_indicators:
                    strong_indicators[file] = True
                elif file in weak_indicators:
                    weak_indicators[file] += 1
        
        strong_count = sum(1 for v in strong_indicators.values() if v)
        weak_count = sum(1 for v in weak_indicators.values() if v >= 2)
        
        is_django = False
        
        if strong_count >= 2:
            is_django = True
            logger.info(f"Proyecto Django detectado por indicadores fuertes: {strong_count}")
        elif strong_count >= 1 and weak_count >= 2:
            is_django = True
            logger.info(f"Proyecto Django detectado por combinación: 1 fuerte + {weak_count} débiles")
        
        if is_django:
            logger.info(f"  Indicadores fuertes: {[k for k, v in strong_indicators.items() if v]}")
            logger.info(f"  Indicadores débiles: {[k for k, v in weak_indicators.items() if v >= 2]}")
        
        return is_django
    
    def _should_ignore_file(self, file_path: Path) -> bool:
        """Determina si un archivo debe ser ignorado."""
        file_name = file_path.name
        file_str = str(file_path)
        
        for ignore_dir in self.ignore_dirs:
            if ignore_dir in file_path.parts:
                logger.debug(f"Ignorando archivo en directorio {ignore_dir}: {file_name}")
                return True
        
        if file_name in self.ignore_files:
            logger.debug(f"Ignorando archivo por nombre: {file_name}")
            return True
        
        for pattern in self.ignore_patterns:
            if pattern.startswith('*') and file_name.endswith(pattern[1:]):
                logger.debug(f"Ignorando archivo por patrón {pattern}: {file_name}")
                return True
            elif pattern.endswith('*') and file_name.startswith(pattern[:-1]):
                logger.debug(f"Ignorando archivo por patrón {pattern}: {file_name}")
                return True
            elif pattern in file_str:
                logger.debug(f"Ignorando archivo por patrón {pattern}: {file_name}")
                return True
        
        return False
    
    def _is_valid_file(self, file_path: Path) -> bool:
        """Verifica si un archivo es válido para indexación."""
        ext = file_path.suffix.lower()
        
        if ext not in self.parsers:
            return False
        
        if self._should_ignore_file(file_path):
            return False
        
        try:
            line_count = sum(1 for _ in open(file_path, 'r', encoding='utf-8', errors='ignore'))
            if line_count > 2000:
                logger.debug(f"Archivo ignorado por líneas: {file_path.name} ({line_count})")
                return False
        except Exception:
            pass
        
        return True
    
    def load_from_zip(self, zip_path: Union[str, Path]) -> Optional[Repository]:
        """
        Carga repositorio desde ZIP.
        Verifica API antes de guardar.
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            logger.error(f"Archivo ZIP no encontrado: {zip_path}")
            return None
        
        logger.info(f"Procesando ZIP: {zip_path.name}")
        
        try:
            repo_name = zip_path.stem
            safe_name = self._sanitize_name(repo_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = self.repositories_dir / f"{safe_name}_{timestamp}"
            temp_dir.mkdir(parents=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            final_path = self._detect_repo_root(temp_dir)
            logger.info(f"Raíz del repositorio: {final_path}")
            
            # Verificar si ya existe por la ruta
            existing = self.db.get_repository_by_path(str(final_path))
            if existing:
                logger.info(f"Repositorio ya existe en BD con ID: {existing['id']}")
                shutil.rmtree(temp_dir)
                return self.load_repository_from_db(existing['id'])
            
            # ANALIZAR sin guardar primero
            repository = self._analyze_directory(final_path, repo_name)
            
            if not repository or not repository.files:
                logger.error("No se encontraron archivos válidos")
                shutil.rmtree(temp_dir)
                return None
            
            # VERIFICAR API KEY ANTES DE GUARDAR
            from application.services.service_factory import ServiceFactory
            if not ServiceFactory.is_api_key_valid():
                logger.warning("API Key no configurada - repositorio no guardado")
                repository._skip_save = True
                return repository
            
            # Guardar en BD solo si API está disponible
            repo_id = self.db.save_repository(repository)
            repository.db_id = repo_id
            logger.info(f"Repositorio guardado en BD con ID: {repo_id}")
            
            return repository
                
        except Exception as e:
            logger.error(f"Error procesando ZIP: {e}")
            if 'temp_dir' in locals() and temp_dir.exists():
                shutil.rmtree(temp_dir)
            return None
    
    def load_from_directory(self, directory_path: Union[str, Path]) -> Optional[Repository]:
        """
        Carga repositorio desde directorio local.
        Verifica API antes de guardar.
        """
        directory_path = Path(directory_path)
        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directorio no válido: {directory_path}")
            return None
        
        # Verificar si el repositorio ya existe en la base de datos
        existing = self.db.get_repository_by_path(str(directory_path))
        if existing:
            logger.info(f"Repositorio ya existe en BD con ID: {existing['id']}")
            return self.load_repository_from_db(existing['id'])
        
        repo_name = directory_path.name
        logger.info(f"Cargando desde directorio: {directory_path}")
        
        # ANALIZAR
        repository = self._analyze_directory(directory_path, repo_name)
        
        if not repository or not repository.files:
            logger.error("No se encontraron archivos válidos")
            return None
        
        # VERIFICAR API KEY ANTES DE GUARDAR
        from application.services.service_factory import ServiceFactory
        if not ServiceFactory.is_api_key_valid():
            logger.warning("API Key no configurada - repositorio no guardado")
            repository._skip_save = True
            return repository
        
        # Guardar en BD solo si API está disponible
        repo_id = self.db.save_repository(repository)
        repository.db_id = repo_id
        logger.info(f"Repositorio guardado en BD con ID: {repo_id}")
        
        return repository
    
    def load_repository_from_db(self, repo_id: int) -> Optional[Repository]:
        """
        Carga un repositorio completo desde la base de datos y archivos en disco.
        """
        try:
            repo_data = self.db.get_repository(repo_id)
            if not repo_data:
                logger.error(f"Repositorio no encontrado: {repo_id}")
                return None
            
            repo_path = Path(repo_data['path'])
            if not repo_path.exists():
                logger.error(f"Ruta de repositorio no existe: {repo_path}")
                return None
            
            repository = Repository(
                name=repo_data['name'],
                path=repo_path,
                language=repo_data['language'],
                file_count=repo_data['file_count'],
                total_lines=repo_data['total_lines'],
                created_at=repo_data['created_at'],
                last_analyzed=repo_data['last_analyzed']
            )
            repository.db_id = repo_id
            
            files_data = self.db.get_files(repo_id)
            
            for file_data in files_data:
                file_path = repo_path / file_data['file_path']
                
                if not file_path.exists():
                    logger.debug(f"Archivo no encontrado en disco: {file_path}")
                    continue
                
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                except Exception as e:
                    logger.error(f"Error leyendo archivo {file_path}: {e}")
                    continue
                
                code_file = CodeFile(
                    path=file_path,
                    extension=file_data['extension'],
                    line_count=file_data['line_count'],
                    content_hash=file_data['content_hash']
                )
                code_file.relative_path = file_data['file_path']
                
                functions = self.db.get_functions(file_data['id'])
                for func_data in functions:
                    code_file.functions.append(Function(
                        name=func_data['name'],
                        line_start=func_data['line_start'],
                        line_end=func_data['line_end'],
                        docstring=func_data['docstring'],
                        complexity=func_data['complexity']
                    ))
                
                classes = self.db.get_classes(file_data['id'])
                for cls_data in classes:
                    class_obj = Class(
                        name=cls_data['name'],
                        line_start=cls_data['line_start'],
                        line_end=cls_data['line_end'],
                        docstring=cls_data['docstring'],
                        parent_class=cls_data['parent_class']
                    )
                    code_file.classes.append(class_obj)
                
                repository.add_file(code_file)
            
            logger.info(f"Repositorio {repository.name} reconstruido desde BD con {len(repository.files)} archivos")
            return repository
            
        except Exception as e:
            logger.error(f"Error cargando repositorio desde BD: {e}")
            return None
    
    def _detect_repo_root(self, extracted_path: Path) -> Path:
        """Detecta la raíz real del repositorio."""
        items = list(extracted_path.iterdir())
        
        if len(items) == 1 and items[0].is_dir():
            inner_dir = items[0]
            if any(f.is_file() for f in inner_dir.iterdir()):
                return inner_dir
            for subdir in inner_dir.iterdir():
                if subdir.is_dir() and any(f.is_file() for f in subdir.iterdir()):
                    return subdir
            return inner_dir
        
        return extracted_path
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitiza nombre para sistema de archivos."""
        safe = re.sub(r'[^\w\-_\.]', '_', name)
        return safe[:100]
    
    def _analyze_directory(self, directory_path: Path, repo_name: str) -> Repository:
        """
        Analiza directorio con procesamiento optimizado.
        
        Args:
            directory_path: Ruta al directorio
            repo_name: Nombre del repositorio
            
        Returns:
            Repositorio analizado
        """
        logger.info(f"Analizando directorio: {directory_path}")
        
        is_django = self._is_django_project(directory_path)
        
        repository = Repository(
            name=repo_name,
            path=directory_path,
            language="django" if is_django else "multi"
        )
        
        if is_django:
            repository.metadata = {
                'framework': 'django',
                'detected_at': datetime.now().isoformat()
            }
            self.stats['django_projects'] += 1
        
        files_by_extension = self._scan_files_by_extension(directory_path)
        
        total_files = sum(len(files) for files in files_by_extension.values())
        logger.info(f"Archivos encontrados por lenguaje: {total_files}")
        
        files_parsed = 0
        
        for ext, files in files_by_extension.items():
            parser = self.parsers.get(ext)
            if not parser:
                continue
            
            logger.info(f"Procesando {len(files)} archivos {ext}")
            
            for file_path in files:
                try:
                    if not self._is_valid_file(file_path):
                        continue
                    
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    parsed = parser.parse_file(file_path, content)
                    
                    if parsed:
                        code_file = self._create_code_file(file_path, content, parsed, directory_path)
                        repository.add_file(code_file)
                        files_parsed += 1
                        
                except Exception as e:
                    logger.error(f"Error parseando {file_path}: {e}")
        
        logger.info(f"Archivos parseados: {files_parsed}/{total_files}")
        
        if is_django:
            models_count = len([f for f in repository.files if 'models.py' in f.name])
            views_count = len([f for f in repository.files if 'views.py' in f.name])
            urls_count = len([f for f in repository.files if 'urls.py' in f.name])
            
            logger.info(f"Proyecto Django detectado:")
            logger.info(f"  Modelos: {models_count}")
            logger.info(f"  Vistas: {views_count}")
            logger.info(f"  URLs: {urls_count}")
            logger.info(f"  Migraciones ignoradas: si")
        
        return repository
    
    def _scan_files_by_extension(self, directory: Path) -> Dict[str, List[Path]]:
        """Escanea archivos agrupados por extensión."""
        files_by_ext = {}
        
        for ext in self.parsers.keys():
            files_by_ext[ext] = []
        
        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue
            
            if self._should_ignore_file(file_path):
                continue
            
            ext = file_path.suffix.lower()
            if ext in files_by_ext:
                files_by_ext[ext].append(file_path)
        
        return files_by_ext
    
    def _create_code_file(self, file_path: Path, content: str, parsed: Dict[str, Any], base_dir: Path) -> CodeFile:
        """Crea objeto CodeFile a partir de datos parseados."""
        code_file = CodeFile(
            path=file_path,
            extension=file_path.suffix,
            line_count=len(content.splitlines()),
            content_hash=hashlib.sha256(content.encode()).hexdigest()[:16]
        )
        
        try:
            rel_path = file_path.relative_to(base_dir)
            code_file.relative_path = str(rel_path)
        except ValueError:
            code_file.relative_path = file_path.name
        
        if 'functions' in parsed:
            for func in parsed['functions']:
                try:
                    code_file.functions.append(Function(**func))
                except Exception as e:
                    logger.debug(f"Error creando función {func.get('name', 'unknown')}: {e}")
        
        if 'classes' in parsed:
            for cls in parsed['classes']:
                try:
                    code_file.classes.append(Class(**cls))
                except Exception as e:
                    logger.debug(f"Error creando clase {cls.get('name', 'unknown')}: {e}")
        
        if 'imports' in parsed:
            code_file.imports = parsed['imports']
        
        return code_file
    
    def get_file_content(self, repo_id: int, file_path: str) -> Optional[str]:
        """Obtiene contenido de un archivo usando caché."""
        content = self.cache.get_text(repo_id, file_path)
        
        if content:
            return content
        
        repo_data = self.db.get_repository(repo_id)
        if not repo_data:
            return None
        
        base_path = Path(repo_data['path'])
        full_path = base_path / file_path
        
        if full_path.exists():
            try:
                content = full_path.read_text(encoding='utf-8')
                self.cache.put_text(repo_id, file_path, content)
                return content
            except Exception as e:
                logger.error(f"Error leyendo archivo: {e}")
        
        return None
    
    def get_repository_path(self, repo_id: int) -> Optional[Path]:
        """Obtiene la ruta física de un repositorio."""
        repo_data = self.db.get_repository(repo_id)
        if repo_data and 'path' in repo_data:
            path = Path(repo_data['path'])
            if path.exists():
                return path
        return None
    
    def get_repository_summary(self, repo_id: int) -> Optional[Dict[str, Any]]:
        """Obtiene un resumen del repositorio por ID."""
        repo_data = self.db.get_repository(repo_id)
        if not repo_data:
            return None
        
        return {
            'id': repo_data['id'],
            'name': repo_data['name'],
            'path': repo_data['path'],
            'language': repo_data['language'],
            'file_count': repo_data['file_count'],
            'total_lines': repo_data['total_lines'],
            'created_at': repo_data['created_at'],
            'last_analyzed': repo_data['last_analyzed']
        }
    
    def list_repositories(self) -> List[Dict[str, Any]]:
        """Lista todos los repositorios analizados."""
        return self.db.list_repositories()
    
    def _delete_directory_with_retry(self, path: Path, max_retries: int = 3) -> bool:
        """
        Elimina un directorio manejando archivos de solo lectura.
        Solo se usa para eliminar copias en data/repositories/.
        """
        if not path.exists():
            return True
        
        if "data/repositories" not in str(path):
            logger.warning(f"No se elimina por seguridad: {path}")
            return False
        
        def _on_rmtree_error(func, path, exc_info):
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception as e:
                logger.warning(f"No se pudo eliminar {path}: {e}")
        
        for attempt in range(max_retries):
            try:
                shutil.rmtree(path, onerror=_on_rmtree_error)
                logger.info(f"Copia del repositorio eliminada: {path}")
                return True
            except Exception as e:
                logger.warning(f"Intento {attempt + 1} falló: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        return False
    
    def delete_repository(self, repo_id: int, delete_files: bool = True) -> bool:
        """
        Elimina repositorio (BD, vectores, caché, pero NO archivos originales del usuario).
        """
        try:
            repo_data = self.db.get_repository(repo_id)
            if not repo_data:
                logger.error(f"Repositorio no encontrado: {repo_id}")
                return False
            
            repo_path = Path(repo_data['path'])
            repo_name = repo_data['name']
            
            result = self.db.delete_repository(repo_id)
            
            if not result:
                logger.error(f"Error eliminando repositorio de BD: {repo_id}")
                return False
            
            try:
                safe_name = repo_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                index_path = Path(f"data/vectors/{safe_name}.index")
                if index_path.exists():
                    index_path.unlink()
                    logger.info(f"Vectores eliminados: {index_path}")
            except Exception as e:
                logger.warning(f"Error eliminando vectores: {e}")
            
            try:
                cache_files_dir = Path("data/cache/files")
                if cache_files_dir.exists():
                    for chunk_file in cache_files_dir.glob(f"{repo_id}:*"):
                        try:
                            chunk_file.unlink()
                        except Exception:
                            pass
                    logger.info(f"Fragmentos de caché eliminados para repo {repo_id}")
            except Exception as e:
                logger.warning(f"Error eliminando fragmentos de caché: {e}")
            
            if delete_files and repo_path and repo_path.exists():
                if "data/repositories" in str(repo_path):
                    logger.info(f"Eliminando copia del repositorio: {repo_path}")
                    self._delete_directory_with_retry(repo_path)
                else:
                    logger.info(f"No se elimina directorio original del usuario: {repo_path}")
            
            logger.info(f"Repositorio {repo_name} eliminado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error eliminando repositorio {repo_id}: {e}")
            return False
    
    def get_repository_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del servicio."""
        cache_stats = self.cache.get_stats()
        
        return {
            'repositories': self.stats,
            'cache': cache_stats,
            'database': {
                'total_repos': len(self.list_repositories())
            }
        }