"""
Servicio de caché LRU para optimizar acceso a archivos y embeddings.
Incluye:
- Archivos (LRU persistente)
- Chunks (texto)
- Embeddings (pickle persistente)
"""

import shutil
import hashlib
import pickle
import time
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
from threading import Lock

logger = logging.getLogger(__name__)


class CacheService:

    def __init__(self, cache_dir: str = "data/cache", max_size_mb: int = 500):
        self.cache_dir = Path(cache_dir)
        self.metadata_dir = self.cache_dir / "metadata"
        self.files_dir = self.cache_dir / "files"
        self.embeddings_dir = self.cache_dir / "embeddings" 

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.lock = Lock()

        self.metadata = self._load_metadata()
        self.access_counts = {}

        logger.info("CacheService inicializado")
        logger.info(f"Directorio: {self.cache_dir}")

    # -------------------------
    # UTILIDADES
    # -------------------------
    def _sanitize_chunk_id(self, chunk_id: str) -> str:
        safe_id = chunk_id.replace(':', '_')
        safe_id = re.sub(r'[<>:"/\\|?*]', '_', safe_id)

        if len(safe_id) > 200:
            safe_id = hashlib.md5(safe_id.encode()).hexdigest()

        return safe_id

    def _load_metadata(self) -> Dict[str, Any]:
        metadata_file = self.metadata_dir / "cache_metadata.json"

        default = {
            'files': {},
            'total_size': 0,
            'created_at': datetime.now().isoformat(),
            'last_cleanup': None,
            'cleanup_count': 0
        }

        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass

        return default

    def _save_metadata(self):
        metadata_file = self.metadata_dir / "cache_metadata.json"

        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error guardando metadata: {e}")

    def _get_file_key(self, repo_id: int, file_path: str) -> str:
        return hashlib.sha256(f"{repo_id}:{file_path}".encode()).hexdigest()[:16]

    # -------------------------
    # ARCHIVOS
    # -------------------------
    def get(self, repo_id: int, file_path: str) -> Optional[Path]:
        key = self._get_file_key(repo_id, file_path)

        with self.lock:
            if key in self.metadata['files']:
                file_path_cache = self.files_dir / key

                if file_path_cache.exists():
                    meta = self.metadata['files'][key]
                    meta['last_access'] = time.time()
                    meta['hits'] = meta.get('hits', 0) + 1
                    return file_path_cache

        return None

    def put(self, repo_id: int, file_path: str, content: bytes) -> Path:
        key = self._get_file_key(repo_id, file_path)
        file = self.files_dir / key

        with self.lock:
            file.write_bytes(content)

            self.metadata['files'][key] = {
                'path': file_path,
                'repo_id': repo_id,
                'size': len(content),
                'last_access': time.time(),
                'created_at': time.time(),
                'hits': 1
            }

            self.metadata['total_size'] += len(content)
            self._save_metadata()

        return file

    # -------------------------
    # CHUNKS
    # -------------------------
    def put_chunk(self, chunk_id: str, content: str):
        safe_id = self._sanitize_chunk_id(chunk_id)
        file = self.files_dir / safe_id

        try:
            file.write_text(content, encoding='utf-8')
        except Exception as e:
            logger.error(f"Error guardando chunk: {e}")

    def get_chunk(self, chunk_id: str) -> Optional[str]:
        safe_id = self._sanitize_chunk_id(chunk_id)
        file = self.files_dir / safe_id

        if file.exists():
            try:
                return file.read_text(encoding='utf-8')
            except Exception:
                pass

        return None

    # -------------------------
    # EMBEDDINGS (NUEVO)
    # -------------------------
    def put_embedding(self, chunk_id: str, vector):
        try:
            safe_id = self._sanitize_chunk_id(chunk_id)
            file = self.embeddings_dir / f"{safe_id}.pkl"

            with open(file, "wb") as f:
                pickle.dump(vector, f)

        except Exception as e:
            logger.error(f"Error guardando embedding: {e}")

    def get_embedding(self, chunk_id: str):
        try:
            safe_id = self._sanitize_chunk_id(chunk_id)
            file = self.embeddings_dir / f"{safe_id}.pkl"

            if file.exists():
                with open(file, "rb") as f:
                    return pickle.load(f)

        except Exception as e:
            logger.error(f"Error leyendo embedding: {e}")

        return None

    # -------------------------
    # LIMPIEZA
    # -------------------------
    def clear_all(self):
        with self.lock:
            shutil.rmtree(self.cache_dir)

            self.cache_dir.mkdir(parents=True)
            self.metadata_dir.mkdir()
            self.files_dir.mkdir()
            self.embeddings_dir.mkdir()

            self.metadata = {
                'files': {},
                'total_size': 0,
                'created_at': datetime.now().isoformat(),
                'last_cleanup': time.time(),
                'cleanup_count': 0
            }

            self._save_metadata()

        logger.info("Cache limpiado completamente")

    # -------------------------
    # STATS
    # -------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            'files': len(self.metadata['files']),
            'size_mb': round(self.metadata['total_size'] / (1024 * 1024), 2),
            'cache_dir': str(self.cache_dir)
        }