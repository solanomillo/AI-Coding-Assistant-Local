"""
Servicio profesional para análisis y persistencia de repositorios.

ARQUITECTURA CORREGIDA - FASE 2/3:
- Los archivos se guardan permanentemente en data/repositories/
- Los metadatos van a MySQL
- No se usan directorios temporales
- Rutas persistentes para indexación posterior
"""

import zipfile
import shutil
from pathlib import Path
from typing import List, Optional, Union
import logging
from datetime import datetime
import re

from domain.models.repository import Repository, CodeFile
from infrastructure.parsers import PythonParser
from infrastructure.database.repository_repo import MySQLRepository

logger = logging.getLogger(__name__)


class RepositoryService:
    """
    Servicio profesional para análisis de repositorios.
    
    RESPONSABILIDADES:
    1. Extraer ZIP a ubicación permanente
    2. Analizar estructura del código con AST
    3. Guardar metadatos en MySQL
    4. Mantener archivos físicos para referencia futura
    
    PRINCIPIOS SOLID:
    - Single Responsibility: Solo análisis y persistencia de repositorios
    - Open/Closed: Fácil de extender con nuevos parsers
    - Dependency Inversion: Depende de abstracciones (MySQLRepository)
    """
    
    def __init__(self):
        """Inicializa el servicio con persistencia permanente."""
        self.db = MySQLRepository()
        self.parsers = {
            '.py': PythonParser()
        }
        self.supported_extensions = {'.py'}
        
        # 📁 Directorios permanentes (NO temporales)
        self.repositories_dir = Path("data/repositories")
        self.repositories_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 50)
        logger.info("✅ RepositoryService inicializado correctamente")
        logger.info(f"📁 Repositorios persistentes en: {self.repositories_dir.absolute()}")
        logger.info("=" * 50)
    
    def load_from_zip(self, zip_path: Union[str, Path]) -> Optional[Repository]:
        """
        Carga repositorio desde ZIP y lo guarda permanentemente.
        VERSIÓN SIMPLIFICADA para depuración.
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            logger.error(f"❌ Archivo ZIP no encontrado: {zip_path}")
            return None
        
        logger.info(f"📦 Procesando ZIP: {zip_path.name}")
        logger.info(f"📦 Tamaño: {zip_path.stat().st_size / 1024:.2f} KB")
        
        try:
            # 1. Crear nombre único para el repositorio
            repo_name = zip_path.stem
            safe_name = self._sanitize_name(repo_name)
            
            # 2. Crear directorio permanente
            repo_dir = self.repositories_dir / safe_name
            counter = 1
            while repo_dir.exists():
                repo_dir = self.repositories_dir / f"{safe_name}_{counter}"
                counter += 1
            
            repo_dir.mkdir(parents=True)
            logger.info(f"📁 Directorio permanente creado: {repo_dir}")
            
            # 3. Extraer ZIP al directorio permanente
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(repo_dir)
            
            # 4. LISTAR contenido para depuración
            logger.info("📋 Contenido del ZIP extraído:")
            for item in repo_dir.iterdir():
                logger.info(f"   - {item.name} {'📁' if item.is_dir() else '📄'}")
            
            # 5. SIMPLIFICAR: Usar el directorio de extracción directamente
            # Sin detección compleja de estructura
            final_repo_path = repo_dir
            logger.info(f"📁 Usando ruta: {final_repo_path}")
            
            # 6. Verificar que hay archivos .py
            py_files = list(final_repo_path.rglob("*.py"))
            logger.info(f"🐍 Archivos Python encontrados: {len(py_files)}")
            
            if not py_files:
                logger.warning("⚠️ No se encontraron archivos .py en el ZIP")
                # Buscar en subdirectorios
                for subdir in final_repo_path.iterdir():
                    if subdir.is_dir():
                        sub_py = list(subdir.rglob("*.py"))
                        if sub_py:
                            logger.info(f"✅ Encontrados {len(sub_py)} archivos .py en {subdir.name}")
                            final_repo_path = subdir
                            break
            
            # 7. Analizar el repositorio
            logger.info(f"🔍 Analizando repositorio en: {final_repo_path}")
            repository = self._analyze_directory(final_repo_path, repo_name)
            
            if repository and repository.files:
                logger.info(f"✅ Repositorio '{repo_name}' procesado exitosamente")
                logger.info(f"   📁 Ruta: {final_repo_path}")
                logger.info(f"   📄 Archivos: {len(repository.files)}")
                return repository
            else:
                logger.error("❌ No se pudo analizar ningún archivo")
                return None
            
        except zipfile.BadZipFile as e:
            logger.error(f"❌ ZIP corrupto: {e}")
            if 'repo_dir' in locals() and repo_dir.exists():
                import shutil
                shutil.rmtree(repo_dir)
            return None
        except Exception as e:
            logger.error(f"❌ Error cargando ZIP: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if 'repo_dir' in locals() and repo_dir.exists():
                import shutil
                shutil.rmtree(repo_dir)
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
            logger.error(f"❌ Directorio no válido: {directory_path}")
            return None
        
        logger.info(f"📁 Cargando desde directorio: {directory_path}")
        
        repo_name = directory_path.name
        repository = self._analyze_directory(directory_path, repo_name)
        
        logger.info(f"✅ Repositorio '{repo_name}' analizado desde directorio")
        
        return repository
    
    def _detect_repo_root(self, extracted_path: Path) -> Path:
        """
        Detecta la raíz real del repositorio (maneja doble anidamiento).
        
        Args:
            extracted_path: Ruta donde se extrajo el ZIP
            
        Returns:
            Ruta real de la raíz del repositorio
        """
        items = list(extracted_path.iterdir())
        
        # Caso 1: Un solo directorio (posible doble anidamiento)
        if len(items) == 1 and items[0].is_dir():
            inner_dir = items[0]
            
            # Verificar si contiene archivos .py directamente
            if list(inner_dir.glob("*.py")):
                logger.debug(f"Estructura simple: {inner_dir}")
                return inner_dir
            
            # Verificar si hay subdirectorios con código
            subdirs = [d for d in inner_dir.iterdir() if d.is_dir()]
            for subdir in subdirs:
                if list(subdir.glob("*.py")):
                    logger.debug(f"Estructura anidada: {subdir}")
                    return subdir
            
            return inner_dir
        
        # Caso 2: Múltiples archivos/directorios en raíz
        logger.debug(f"Estructura plana: {extracted_path}")
        return extracted_path
    
    def _sanitize_name(self, name: str) -> str:
        """
        Sanitiza nombre para usar como directorio.
        
        Args:
            name: Nombre original
            
        Returns:
            Nombre seguro para sistema de archivos
        """
        # Reemplazar caracteres no seguros
        safe = re.sub(r'[^\w\-_\. ]', '_', name)
        safe = safe.replace(' ', '_')
        return safe[:100]  # Limitar longitud
    
    def _analyze_directory(self, directory_path: Path, repo_name: str) -> Optional[Repository]:
        """
        Analiza directorio y construye Repository.
        VERSIÓN CORREGIDA - Establece current_repo_root.
        """
        logger.info(f"🔍 Analizando repositorio: {repo_name}")
        logger.info(f"📁 Ruta física: {directory_path.absolute()}")
        
        # Guardar la raíz del repositorio para rutas relativas
        self.current_repo_root = directory_path
        
        # Verificar que el directorio existe
        if not directory_path.exists():
            logger.error(f"❌ El directorio no existe: {directory_path}")
            return None
        
        # Crear repositorio
        repository = Repository(
            name=repo_name,
            path=directory_path,
            language="python"
        )
        
        # Buscar archivos Python
        python_files = []
        for file_path in directory_path.rglob("*.py"):
            # Ignorar directorios comunes
            if any(ignore in str(file_path) for ignore in ['__pycache__', 'venv', 'env', '.venv']):
                continue
            python_files.append(file_path)
        
        logger.info(f"📄 Encontrados {len(python_files)} archivos Python")
        
        if not python_files:
            logger.warning("⚠️ No se encontraron archivos Python")
            return repository
        
        # Parsear cada archivo
        files_parsed = 0
        for file_path in python_files:
            try:
                rel_path = file_path.relative_to(directory_path)
                logger.debug(f"  Parseando: {rel_path}")
                code_file = self._parse_file(file_path)
                if code_file:
                    repository.add_file(code_file)
                    files_parsed += 1
                    logger.debug(f"  ✅ {rel_path}")
            except Exception as e:
                logger.error(f"  ❌ Error parseando {file_path.name}: {e}")
        
        # Actualizar timestamp
        repository.last_analyzed = datetime.now()
        
        # Guardar en base de datos
        try:
            repo_id = self.db.save_repository(repository)
            logger.info(f"💾 Metadatos guardados en MySQL con ID: {repo_id}")
        except Exception as e:
            logger.error(f"❌ Error guardando en BD: {e}")
        
        logger.info(f"📊 Resumen: {files_parsed} archivos parseados de {len(python_files)}")
        
        # Limpiar variable temporal
        self.current_repo_root = None
        
        return repository
    
    def _scan_python_files(self, directory: Path) -> List[Path]:
        """
        Escanea recursivamente archivos Python.
        
        Args:
            directory: Directorio a escanear
            
        Returns:
            Lista de rutas de archivos Python
        """
        python_files = []
        
        # Directorios a ignorar (comunes en proyectos)
        ignore_dirs = {
            'venv', 'env', '.venv', '__pycache__', 
            'node_modules', '.git', '.idea', '.vscode',
            'dist', 'build', '*.egg-info'
        }
        
        for file_path in directory.rglob("*.py"):
            # Verificar si está en directorio ignorado
            if any(ignore in file_path.parts for ignore in ignore_dirs):
                continue
            python_files.append(file_path)
        
        return python_files
    
    def _parse_file(self, file_path: Path) -> Optional[CodeFile]:
        """
        Parsea un archivo y guarda la ruta relativa correcta.
        VERSIÓN CORREGIDA - Usa el setter ahora disponible.
        """
        parser = self.parsers.get('.py')
        if not parser:
            return None
        
        try:
            # Intentar diferentes codificaciones
            encodings = ['utf-8', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                logger.warning(f"No se pudo leer {file_path} con ninguna codificación")
                return None
            
            # Parsear el archivo
            code_file = parser.parse_file(file_path, content)
            
            if code_file:
                # 🔥 CORRECCIÓN: Establecer ruta relativa usando el setter
                # Intentar obtener ruta relativa al repositorio
                if hasattr(self, 'current_repo_root') and self.current_repo_root:
                    try:
                        rel_path = str(file_path.relative_to(self.current_repo_root))
                        code_file.relative_path = rel_path  # Ahora funciona con el setter
                        logger.debug(f"Ruta relativa establecida: {rel_path}")
                    except ValueError:
                        # Si no se puede, usar el nombre del archivo
                        code_file.relative_path = file_path.name
                        logger.debug(f"Usando nombre de archivo como ruta: {file_path.name}")
                else:
                    # Si no hay repo root, usar la ruta completa
                    code_file.relative_path = str(file_path)
            
            return code_file
            
        except Exception as e:
            logger.error(f"Error parseando {file_path}: {e}")
            return None
    
    def _detect_language(self, directory: Path) -> str:
        """
        Detecta el lenguaje principal del repositorio.
        
        Args:
            directory: Directorio del repositorio
            
        Returns:
            Nombre del lenguaje detectado
        """
        if self._scan_python_files(directory):
            return "python"
        return "unknown"
    
    def get_repository_path(self, repo_id: int) -> Optional[Path]:
        """
        Obtiene la ruta física de un repositorio por ID.
        
        Args:
            repo_id: ID del repositorio
            
        Returns:
            Path al repositorio o None
        """
        repo_data = self.db.get_repository(repo_id)
        if repo_data and 'path' in repo_data:
            path = Path(repo_data['path'])
            if path.exists():
                return path
            else:
                logger.warning(f"⚠️ Ruta no existe en disco: {path}")
        return None
    
    def delete_repository(self, repo_id: int, delete_files: bool = True) -> bool:
        """
        Elimina repositorio (BD y opcionalmente archivos).
        
        Args:
            repo_id: ID del repositorio
            delete_files: Si True, elimina también archivos físicos
            
        Returns:
            True si éxito
        """
        try:
            # Obtener ruta antes de eliminar de BD
            repo_path = self.get_repository_path(repo_id)
            
            # Eliminar de BD
            result = self.db.delete_repository(repo_id)
            
            # Eliminar archivos si se solicita
            if delete_files and repo_path and repo_path.exists():
                shutil.rmtree(repo_path)
                logger.info(f"🗑️ Archivos eliminados: {repo_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error eliminando repositorio {repo_id}: {e}")
            return False
    
    def list_repositories(self) -> List[dict]:
        """
        Lista todos los repositorios con información de estado.
        
        Returns:
            Lista de repositorios con metadatos
        """
        repos = self.db.list_repositories()
        
        # Enriquecer con información de archivos
        for repo in repos:
            path = Path(repo['path'])
            repo['files_exist'] = path.exists()
            if repo['files_exist']:
                # Calcular tamaño total
                total_size = sum(f.stat().st_size for f in path.rglob('*') 
                               if f.is_file()) / (1024 * 1024)
                repo['size_mb'] = round(total_size, 2)
        
        return repos