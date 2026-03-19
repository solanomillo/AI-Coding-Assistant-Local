"""
Servicio de caché LRU para optimizar acceso a archivos.
Implementa caché con auto-limpieza y persistencia de metadatos.
"""

import os
import shutil
import hashlib
import pickle
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, BinaryIO
from datetime import datetime, timedelta
import logging
from threading import Lock

logger = logging.getLogger(__name__)


class CacheService:
    """
    Servicio de caché LRU para archivos de repositorios.
    
    Características:
    - Almacena solo archivos recientemente usados
    - Auto-limpieza cuando excede tamaño máximo
    - Persistencia de metadatos
    - Thread-safe para operaciones concurrentes
    - Estadísticas de uso
    """
    
    def __init__(self, cache_dir: str = "data/cache", max_size_mb: int = 500):
        """
        Inicializa el servicio de caché.
        
        Args:
            cache_dir: Directorio para almacenar caché
            max_size_mb: Tamaño máximo del caché en MB
        """
        self.cache_dir = Path(cache_dir)
        self.metadata_dir = self.cache_dir / "metadata"
        self.files_dir = self.cache_dir / "files"
        
        # Crear directorios
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.lock = Lock()
        
        # Metadatos del caché
        self.metadata = self._load_metadata()
        self.access_counts = {}  # Contador de accesos para estadísticas
        
        logger.info(f"Servicio de caché inicializado")
        logger.info(f"  Directorio: {self.cache_dir}")
        logger.info(f"  Tamaño máximo: {max_size_mb} MB")
        logger.info(f"  Archivos en caché: {len(self.metadata['files'])}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Carga metadatos del caché desde disco.
        
        Returns:
            Diccionario con metadatos
        """
        metadata_file = self.metadata_dir / "cache_metadata.json"
        
        default_metadata = {
            'files': {},  # file_key -> {path, size, last_access, hits, created_at}
            'total_size': 0,
            'created_at': datetime.now().isoformat(),
            'last_cleanup': None,
            'cleanup_count': 0
        }
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                logger.debug(f"Metadatos cargados: {len(metadata['files'])} archivos")
                return metadata
            except Exception as e:
                logger.error(f"Error cargando metadatos: {e}")
        
        return default_metadata
    
    def _save_metadata(self) -> None:
        """Guarda metadatos en disco."""
        metadata_file = self.metadata_dir / "cache_metadata.json"
        
        try:
            # Crear backup antes de guardar
            if metadata_file.exists():
                backup = self.metadata_dir / "cache_metadata.backup.json"
                shutil.copy2(metadata_file, backup)
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            
            logger.debug("Metadatos guardados correctamente")
            
        except Exception as e:
            logger.error(f"Error guardando metadatos: {e}")
    
    def _get_file_key(self, repo_id: int, file_path: str) -> str:
        """
        Genera clave única para un archivo.
        
        Args:
            repo_id: ID del repositorio
            file_path: Ruta del archivo
            
        Returns:
            Clave única
        """
        unique = f"{repo_id}:{file_path}"
        return hashlib.sha256(unique.encode()).hexdigest()[:16]
    
    def get(self, repo_id: int, file_path: str) -> Optional[Path]:
        """
        Obtiene un archivo del caché.
        
        Args:
            repo_id: ID del repositorio
            file_path: Ruta del archivo
            
        Returns:
            Ruta al archivo en caché o None
        """
        file_key = self._get_file_key(repo_id, file_path)
        
        with self.lock:
            if file_key in self.metadata['files']:
                metadata = self.metadata['files'][file_key]
                cached_file = self.files_dir / file_key
                
                if cached_file.exists():
                    # Actualizar metadatos
                    metadata['last_access'] = time.time()
                    metadata['hits'] = metadata.get('hits', 0) + 1
                    
                    # Actualizar contador de accesos
                    self.access_counts[file_key] = self.access_counts.get(file_key, 0) + 1
                    
                    logger.debug(f"Cache hit: {file_path} (hits: {metadata['hits']})")
                    
                    # Guardar metadatos periódicamente (cada 10 accesos)
                    if metadata['hits'] % 10 == 0:
                        self._save_metadata()
                    
                    return cached_file
                else:
                    # Archivo perdido, limpiar metadata
                    logger.warning(f"Archivo en caché no encontrado en disco: {file_key}")
                    del self.metadata['files'][file_key]
                    self.metadata['total_size'] -= metadata['size']
                    self._save_metadata()
        
        logger.debug(f"Cache miss: {file_path}")
        return None
    
    def put(self, repo_id: int, file_path: str, content: bytes) -> Path:
        """
        Guarda un archivo en caché.
        
        Args:
            repo_id: ID del repositorio
            file_path: Ruta original del archivo
            content: Contenido del archivo
            
        Returns:
            Ruta al archivo en caché
        """
        file_key = self._get_file_key(repo_id, file_path)
        cached_file = self.files_dir / file_key
        file_size = len(content)
        
        with self.lock:
            # Verificar si ya existe
            if file_key in self.metadata['files']:
                old_size = self.metadata['files'][file_key]['size']
                self.metadata['total_size'] -= old_size
            
            # Guardar archivo
            cached_file.write_bytes(content)
            
            # Actualizar metadatos
            self.metadata['files'][file_key] = {
                'path': file_path,
                'repo_id': repo_id,
                'size': file_size,
                'last_access': time.time(),
                'created_at': time.time(),
                'hits': self.metadata['files'].get(file_key, {}).get('hits', 0) + 1
            }
            
            self.metadata['total_size'] += file_size
            
            logger.info(f"Archivo guardado en caché: {file_path} ({file_size} bytes)")
            
            # Limpiar si excede tamaño máximo
            if self.metadata['total_size'] > self.max_size_bytes:
                self._cleanup()
            
            # Guardar metadatos
            self._save_metadata()
        
        return cached_file
    
    def get_text(self, repo_id: int, file_path: str, encoding: str = 'utf-8') -> Optional[str]:
        """
        Obtiene contenido de texto de un archivo.
        
        Args:
            repo_id: ID del repositorio
            file_path: Ruta del archivo
            encoding: Codificación del archivo
            
        Returns:
            Contenido del archivo o None
        """
        cached_file = self.get(repo_id, file_path)
        
        if cached_file:
            try:
                return cached_file.read_text(encoding=encoding)
            except Exception as e:
                logger.error(f"Error leyendo archivo de caché: {e}")
        
        return None
    
    def put_text(self, repo_id: int, file_path: str, content: str, encoding: str = 'utf-8') -> Path:
        """
        Guarda contenido de texto en caché.
        
        Args:
            repo_id: ID del repositorio
            file_path: Ruta original del archivo
            content: Contenido de texto
            encoding: Codificación
            
        Returns:
            Ruta al archivo en caché
        """
        return self.put(repo_id, file_path, content.encode(encoding))
    
    def _cleanup(self) -> None:
        """Limpia archivos antiguos cuando se excede el tamaño máximo."""
        logger.warning(f"Iniciando limpieza de caché. Tamaño actual: {self.metadata['total_size']} bytes")
        
        # Ordenar archivos por última acceso (más antiguos primero)
        sorted_files = sorted(
            self.metadata['files'].items(),
            key=lambda x: x[1]['last_access']
        )
        
        bytes_to_free = self.metadata['total_size'] - self.max_size_bytes
        bytes_freed = 0
        files_removed = 0
        
        for file_key, metadata in sorted_files:
            if bytes_freed >= bytes_to_free:
                break
            
            # Eliminar archivo
            cached_file = self.files_dir / file_key
            if cached_file.exists():
                cached_file.unlink()
            
            bytes_freed += metadata['size']
            del self.metadata['files'][file_key]
            files_removed += 1
            
            logger.debug(f"Eliminado: {metadata['path']} ({metadata['size']} bytes)")
        
        self.metadata['total_size'] -= bytes_freed
        self.metadata['last_cleanup'] = time.time()
        self.metadata['cleanup_count'] = self.metadata.get('cleanup_count', 0) + 1
        
        logger.info(f"Limpieza completada: {files_removed} archivos, {bytes_freed} bytes liberados")
        
        # Guardar metadatos después de limpieza
        self._save_metadata()
    
    def clear_repository(self, repo_id: int) -> None:
        """
        Limpia todos los archivos de un repositorio del caché.
        
        Args:
            repo_id: ID del repositorio
        """
        with self.lock:
            to_delete = []
            
            for file_key, metadata in self.metadata['files'].items():
                if metadata['repo_id'] == repo_id:
                    to_delete.append((file_key, metadata))
            
            for file_key, metadata in to_delete:
                cached_file = self.files_dir / file_key
                if cached_file.exists():
                    cached_file.unlink()
                
                del self.metadata['files'][file_key]
                self.metadata['total_size'] -= metadata['size']
                
                logger.info(f"Eliminado del caché: {metadata['path']}")
            
            self._save_metadata()
        
        logger.info(f"Caché limpiado para repositorio {repo_id}")
    
    def clear_all(self) -> None:
        """Limpia todo el caché."""
        with self.lock:
            # Eliminar archivos
            shutil.rmtree(self.files_dir)
            self.files_dir.mkdir(parents=True)
            
            # Reiniciar metadatos
            self.metadata = {
                'files': {},
                'total_size': 0,
                'created_at': datetime.now().isoformat(),
                'last_cleanup': time.time(),
                'cleanup_count': self.metadata.get('cleanup_count', 0) + 1
            }
            
            self.access_counts = {}
            
            # Guardar
            self._save_metadata()
            
        logger.info("Caché completamente limpiado")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del caché.
        
        Returns:
            Diccionario con estadísticas
        """
        with self.lock:
            # Calcular hits totales
            total_hits = sum(f.get('hits', 0) for f in self.metadata['files'].values())
            
            # Obtener archivos más accedidos
            most_accessed = sorted(
                [
                    {
                        'path': m['path'],
                        'hits': m.get('hits', 0),
                        'size_kb': m['size'] / 1024
                    }
                    for m in self.metadata['files'].values()
                ],
                key=lambda x: x['hits'],
                reverse=True
            )[:10]
            
            return {
                'total_files': len(self.metadata['files']),
                'total_size_mb': round(self.metadata['total_size'] / (1024 * 1024), 2),
                'max_size_mb': round(self.max_size_bytes / (1024 * 1024), 2),
                'usage_percent': round(
                    (self.metadata['total_size'] / self.max_size_bytes) * 100, 2
                ),
                'total_hits': total_hits,
                'avg_hits_per_file': round(total_hits / len(self.metadata['files']), 2) if self.metadata['files'] else 0,
                'cache_dir': str(self.cache_dir),
                'last_cleanup': self.metadata.get('last_cleanup'),
                'cleanup_count': self.metadata.get('cleanup_count', 0),
                'most_accessed': most_accessed
            }
    
    def prefetch(self, repo_id: int, file_paths: List[str], base_dir: Path) -> None:
        """
        Precarga múltiples archivos en caché.
        
        Args:
            repo_id: ID del repositorio
            file_paths: Lista de rutas de archivos
            base_dir: Directorio base donde buscar los archivos
        """
        logger.info(f"Precargando {len(file_paths)} archivos en caché")
        
        for file_path in file_paths:
            try:
                # Verificar si ya está en caché
                if self.get(repo_id, file_path):
                    continue
                
                # Leer archivo
                full_path = base_dir / file_path
                if full_path.exists():
                    content = full_path.read_bytes()
                    self.put(repo_id, file_path, content)
                    
            except Exception as e:
                logger.error(f"Error precargando {file_path}: {e}")
        
        logger.info("Precarga completada")
    
    def __contains__(self, key: tuple) -> bool:
        """
        Verifica si un archivo está en caché.
        
        Args:
            key: Tupla (repo_id, file_path)
            
        Returns:
            True si está en caché
        """
        repo_id, file_path = key
        file_key = self._get_file_key(repo_id, file_path)
        return file_key in self.metadata['files']