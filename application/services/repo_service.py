"""
Servicio optimizado para análisis de repositorios.
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
        """Carga repositorio desde ZIP."""
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
            
            repository = self._analyze_directory(final_path, repo_name)
            
            if repository and repository.files:
                repo_id = self.db.save_repository(repository)
                repository.db_id = repo_id
                
                logger.info(f"Repositorio procesado: {len(repository.files)} archivos")
                return repository
            else:
                logger.error("No se encontraron archivos válidos")
                shutil.rmtree(temp_dir)
                return None
                
        except Exception as e:
            logger.error(f"Error procesando ZIP: {e}")
            if 'temp_dir' in locals() and temp_dir.exists():
                shutil.rmtree(temp_dir)
            return None
    
    def load_from_directory(self, directory_path: Union[str, Path]) -> Optional[Repository]:
        """Carga repositorio desde directorio local."""
        directory_path = Path(directory_path)
        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directorio no válido: {directory_path}")
            return None
        
        repo_name = directory_path.name
        logger.info(f"Cargando desde directorio: {directory_path}")
        
        repository = self._analyze_directory(directory_path, repo_name)
        
        if repository and repository.files:
            repo_id = self.db.save_repository(repository)
            repository.db_id = repo_id
            return repository
        else:
            logger.error("No se encontraron archivos válidos")
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
        """Analiza directorio con procesamiento optimizado."""
        logger.info(f"Analizando directorio: {directory_path}")
        
        repository = Repository(
            name=repo_name,
            path=directory_path,
            language="multi"
        )
        
        files_by_extension = self._scan_files_by_extension(directory_path)
        
        total_files = sum(len(files) for files in files_by_extension.values())
        logger.info(f"Archivos encontrados: {total_files}")
        
        files_parsed = 0
        
        for ext, files in files_by_extension.items():
            parser = self.parsers.get(ext)
            if not parser:
                continue
            
            logger.info(f"Procesando {len(files)} archivos {ext}")
            
            for file_path in files:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    
                    parsed = parser.parse_file(file_path, content)
                    
                    if parsed:
                        code_file = self._create_code_file(file_path, content, parsed, directory_path)
                        repository.add_file(code_file)
                        files_parsed += 1
                        
                except Exception as e:
                    logger.error(f"Error parseando {file_path}: {e}")
        
        logger.info(f"Archivos parseados: {files_parsed}/{total_files}")
        
        return repository
    
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
                code_file.functions.append(Function(**func))
        
        if 'classes' in parsed:
            for cls in parsed['classes']:
                code_file.classes.append(Class(**cls))
        
        if 'imports' in parsed:
            code_file.imports = parsed['imports']
        
        return code_file
    
    def _scan_files_by_extension(self, directory: Path) -> Dict[str, List[Path]]:
        """Escanea archivos agrupados por extensión."""
        files_by_ext = {}
        
        ignore_dirs = {
            'venv', 'env', '.venv', '__pycache__',
            'node_modules', '.git', '.idea', '.vscode',
            'dist', 'build'
        }
        
        for ext in self.parsers.keys():
            files_by_ext[ext] = []
        
        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue
            
            if any(ignore in file_path.parts for ignore in ignore_dirs):
                continue
            
            ext = file_path.suffix.lower()
            if ext in files_by_ext:
                files_by_ext[ext].append(file_path)
        
        return files_by_ext
    
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
    
    def delete_repository(self, repo_id: int, delete_files: bool = True) -> bool:
        """Elimina repositorio."""
        try:
            repo_path = self.get_repository_path(repo_id)
            result = self.db.delete_repository(repo_id)
            
            if delete_files and repo_path and repo_path.exists():
                shutil.rmtree(repo_path)
                logger.info(f"Archivos eliminados: {repo_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error eliminando repositorio: {e}")
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