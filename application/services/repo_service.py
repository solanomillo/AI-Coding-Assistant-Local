"""
Servicio optimizado para análisis de repositorios.
Usa parsers específicos por lenguaje y caché inteligente.
"""

import zipfile
import shutil
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import logging
from datetime import datetime
import re
import hashlib

from domain.models.repository import Repository, CodeFile, Function, Class
from infrastructure.database.mysql_repository import MySQLRepository
from infrastructure.parsers.python_parser import PythonParser
from infrastructure.parsers.javascript_parser import JavaScriptParser
from infrastructure.parsers.html_parser import HTMLParser
from infrastructure.parsers.css_parser import CSSParser
from application.services.cache_service import CacheService

logger = logging.getLogger(__name__)


class RepositoryService:
    """
    Servicio optimizado para análisis de repositorios.
    
    Características:
    - Soporta múltiples lenguajes
    - Usa caché para archivos frecuentes
    - Procesamiento por lotes
    - Integración con MySQL para metadatos
    """
    
    def __init__(self):
        """Inicializa el servicio con parsers multi-lenguaje."""
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
        
        # Estadísticas
        self.stats = {
            'total_repos': 0,
            'total_files': 0,
            'total_functions': 0,
            'total_classes': 0,
            'languages': {}
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
    
    def load_from_zip(self, zip_path: Union[str, Path]) -> Optional[Repository]:
        """
        Carga repositorio desde ZIP con optimizaciones.
        
        Args:
            zip_path: Ruta al archivo ZIP
            
        Returns:
            Repositorio analizado
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            logger.error(f"Archivo ZIP no encontrado: {zip_path}")
            return None
        
        logger.info(f"Procesando ZIP: {zip_path.name}")
        logger.info(f"Tamaño: {zip_path.stat().st_size / 1024:.2f} KB")
        
        try:
            # Crear directorio temporal para extracción
            repo_name = zip_path.stem
            safe_name = self._sanitize_name(repo_name)
            
            # Usar timestamp para evitar conflictos
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = self.repositories_dir / f"{safe_name}_{timestamp}"
            temp_dir.mkdir(parents=True)
            
            # Extraer ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Detectar estructura
            final_path = self._detect_repo_root(temp_dir)
            logger.info(f"Raíz del repositorio: {final_path}")
            
            # Analizar
            repository = self._analyze_directory(final_path, repo_name)
            
            if repository and repository.files:
                # Guardar en base de datos
                repo_id = self.db.save_repository(repository)
                repository.db_id = repo_id
                
                logger.info(f"Repositorio procesado exitosamente:")
                logger.info(f"  ID: {repo_id}")
                logger.info(f"  Archivos: {len(repository.files)}")
                logger.info(f"  Líneas totales: {repository.total_lines}")
                
                # Actualizar estadísticas
                self._update_stats(repository)
                
                return repository
            else:
                logger.error("No se encontraron archivos válidos")
                shutil.rmtree(temp_dir)
                return None
                
        except zipfile.BadZipFile as e:
            logger.error(f"ZIP corrupto: {e}")
            if 'temp_dir' in locals() and temp_dir.exists():
                shutil.rmtree(temp_dir)
            return None
        except Exception as e:
            logger.error(f"Error procesando ZIP: {e}")
            if 'temp_dir' in locals() and temp_dir.exists():
                shutil.rmtree(temp_dir)
            return None
    
    def load_from_directory(self, directory_path: Union[str, Path]) -> Optional[Repository]:
        """
        Carga repositorio desde directorio local.
        
        Args:
            directory_path: Ruta al directorio
            
        Returns:
            Repositorio analizado
        """
        directory_path = Path(directory_path)
        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directorio no válido: {directory_path}")
            return None
        
        repo_name = directory_path.name
        logger.info(f"Cargando desde directorio: {directory_path}")
        
        repository = self._analyze_directory(directory_path, repo_name)
        
        if repository and repository.files:
            # Guardar en base de datos
            repo_id = self.db.save_repository(repository)
            repository.db_id = repo_id
            
            logger.info(f"Repositorio procesado exitosamente:")
            logger.info(f"  ID: {repo_id}")
            logger.info(f"  Archivos: {len(repository.files)}")
            logger.info(f"  Líneas totales: {repository.total_lines}")
            
            # Actualizar estadísticas
            self._update_stats(repository)
            
            return repository
        else:
            logger.error("No se encontraron archivos válidos")
            return None
    
    def _detect_repo_root(self, extracted_path: Path) -> Path:
        """
        Detecta la raíz real del repositorio.
        
        Args:
            extracted_path: Ruta de extracción
            
        Returns:
            Ruta real de la raíz
        """
        items = list(extracted_path.iterdir())
        
        # Si hay un solo directorio, probablemente es la raíz
        if len(items) == 1 and items[0].is_dir():
            inner_dir = items[0]
            
            # Verificar si contiene archivos directamente
            if any(f.is_file() for f in inner_dir.iterdir()):
                return inner_dir
            
            # Buscar subdirectorios con archivos
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
        Analiza directorio con procesamiento optimizado y caché.
        
        Args:
            directory_path: Ruta al directorio
            repo_name: Nombre del repositorio
            
        Returns:
            Repositorio analizado
        """
        logger.info(f"Analizando directorio: {directory_path}")
        
        repository = Repository(
            name=repo_name,
            path=directory_path,
            language="multi"
        )
        
        # Escanear archivos por extensión
        files_by_extension = self._scan_files_by_extension(directory_path)
        
        total_files = sum(len(files) for files in files_by_extension.values())
        logger.info(f"Archivos encontrados por lenguaje: {total_files}")
        
        # Parsear por lotes
        files_parsed = 0
        functions_found = 0
        classes_found = 0
        
        for ext, files in files_by_extension.items():
            parser = self.parsers.get(ext)
            if not parser:
                continue
            
            logger.info(f"Procesando {len(files)} archivos {ext}")
            
            for file_path in files:
                try:
                    # Verificar si el archivo ya está en caché
                    rel_path = str(file_path.relative_to(directory_path))
                    cached_content = self.cache.get_text(repository.db_id if hasattr(repository, 'db_id') else 0, rel_path)
                    
                    if cached_content:
                        # Usar contenido del caché
                        content = cached_content
                        logger.debug(f"Archivo obtenido de caché: {rel_path}")
                    else:
                        # Leer archivo y guardar en caché
                        content = file_path.read_text(encoding='utf-8')
                        
                        # Guardar en caché para futuros usos
                        if hasattr(repository, 'db_id'):
                            self.cache.put_text(repository.db_id, rel_path, content)
                    
                    # Parsear con parser específico
                    parsed = parser.parse_file(file_path, content)
                    
                    if parsed:
                        code_file = self._create_code_file(file_path, content, parsed, directory_path)
                        repository.add_file(code_file)
                        files_parsed += 1
                        
                        functions_found += len(code_file.functions)
                        classes_found += len(code_file.classes)
                        
                except Exception as e:
                    logger.error(f"Error parseando {file_path}: {e}")
        
        logger.info(f"Análisis completado:")
        logger.info(f"  Archivos parseados: {files_parsed}/{total_files}")
        logger.info(f"  Funciones encontradas: {functions_found}")
        logger.info(f"  Clases encontradas: {classes_found}")
        
        return repository
    
    def _create_code_file(self, file_path: Path, content: str, parsed: Dict[str, Any], base_dir: Path) -> CodeFile:
        """
        Crea objeto CodeFile a partir de datos parseados.
        
        Args:
            file_path: Ruta del archivo
            content: Contenido del archivo
            parsed: Datos parseados
            base_dir: Directorio base para ruta relativa
            
        Returns:
            CodeFile listo para usar
        """
        code_file = CodeFile(
            path=file_path,
            extension=file_path.suffix,
            line_count=len(content.splitlines()),
            content_hash=hashlib.sha256(content.encode()).hexdigest()[:16]
        )
        
        # Establecer ruta relativa
        try:
            rel_path = file_path.relative_to(base_dir)
            code_file.relative_path = str(rel_path)
        except ValueError:
            code_file.relative_path = file_path.name
        
        # Agregar funciones
        if 'functions' in parsed:
            for func in parsed['functions']:
                code_file.functions.append(Function(**func))
        
        # Agregar clases
        if 'classes' in parsed:
            for cls in parsed['classes']:
                code_file.classes.append(Class(**cls))
        
        # Agregar imports
        if 'imports' in parsed:
            code_file.imports = parsed['imports']
        
        return code_file
    
    def _scan_files_by_extension(self, directory: Path) -> Dict[str, List[Path]]:
        """
        Escanea archivos agrupados por extensión.
        
        Args:
            directory: Directorio a escanear
            
        Returns:
            Diccionario extensión -> lista de archivos
        """
        files_by_ext = {}
        
        # Directorios a ignorar
        ignore_dirs = {
            'venv', 'env', '.venv', '__pycache__',
            'node_modules', '.git', '.idea', '.vscode',
            'dist', 'build', '*.egg-info'
        }
        
        for ext in self.parsers.keys():
            files_by_ext[ext] = []
        
        # Escaneo optimizado con rglob
        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Ignorar directorios de sistema
            if any(ignore in file_path.parts for ignore in ignore_dirs):
                continue
            
            ext = file_path.suffix.lower()
            if ext in files_by_ext:
                files_by_ext[ext].append(file_path)
        
        return files_by_ext
    
    def _update_stats(self, repository: Repository) -> None:
        """Actualiza estadísticas globales."""
        self.stats['total_repos'] += 1
        self.stats['total_files'] += len(repository.files)
        
        for file in repository.files:
            self.stats['total_functions'] += len(file.functions)
            self.stats['total_classes'] += len(file.classes)
            
            # Estadísticas por lenguaje
            ext = file.extension.lower()
            parser = self.parsers.get(ext)
            if parser:
                lang = parser.language
                if lang not in self.stats['languages']:
                    self.stats['languages'][lang] = 0
                self.stats['languages'][lang] += 1
    
    def get_file_content(self, repo_id: int, file_path: str) -> Optional[str]:
        """
        Obtiene contenido de un archivo usando caché.
        
        Args:
            repo_id: ID del repositorio
            file_path: Ruta del archivo
            
        Returns:
            Contenido del archivo o None
        """
        # Intentar obtener de caché primero
        content = self.cache.get_text(repo_id, file_path)
        
        if content:
            logger.debug(f"Archivo obtenido de caché: {file_path}")
            return content
        
        # Si no está en caché, buscar en disco
        repo_data = self.db.get_repository(repo_id)
        if not repo_data:
            logger.error(f"Repositorio no encontrado: {repo_id}")
            return None
        
        base_path = Path(repo_data['path'])
        full_path = base_path / file_path
        
        if full_path.exists():
            try:
                content = full_path.read_text(encoding='utf-8')
                # Guardar en caché para futuro
                self.cache.put_text(repo_id, file_path, content)
                logger.debug(f"Archivo guardado en caché: {file_path}")
                return content
            except Exception as e:
                logger.error(f"Error leyendo archivo {file_path}: {e}")
        
        return None
    
    def get_repository_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del servicio.
        
        Returns:
            Diccionario con estadísticas
        """
        cache_stats = self.cache.get_stats()
        
        return {
            'repositories': self.stats,
            'cache': cache_stats,
            'database': {
                'total_repos': len(self.db.list_repositories())
            }
        }