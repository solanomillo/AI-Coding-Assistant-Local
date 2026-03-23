"""
Servicio RAG con Gemini y FAISS.
Versión optimizada con control de rate limiting.
"""

import logging
import time
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import threading
import queue
import numpy as np
import traceback
import re

from infrastructure.embeddings.gemini_embedding import GeminiEmbedding
from infrastructure.vector_db.faiss_store import FAISSStore
from infrastructure.llm_clients.gemini_llm import GeminiLLM
from domain.models.repository import Repository
from application.services.cache_service import CacheService

logger = logging.getLogger(__name__)


class RAGService:
    """
    Servicio RAG optimizado con control de rate limiting.
    """
    
    # Constantes optimizadas para free tier
    MAX_CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    MAX_FRAGMENTS_PER_FILE = 10
    MAX_FILE_SIZE_MB = 1
    MAX_FILE_LINES = 2000
    API_TIMEOUT_SECONDS = 30
    BATCH_SIZE = 20
    BATCH_DELAY = 2
    API_RETRY_BASE_DELAY = 60
    
    # Extensiones prioritarias
    PRIORITY_EXTENSIONS = {
        '.py', '.pyx', '.js', '.jsx', '.mjs', '.cjs',
        '.ts', '.tsx', '.html', '.htm', '.css', '.scss', '.sass',
        '.json', '.yaml', '.yml', '.sql', '.sh', '.bash', '.zsh',
        '.go', '.rs', '.java', '.cpp', '.c', '.h', '.hpp', '.rb', '.php', '.vue'
    }
    
    # Archivos a ignorar
    IGNORE_FILES = {
        '.env', '.gitignore', '.dockerignore', '.eslintignore',
        'package-lock.json', 'yarn.lock', 'poetry.lock',
        'requirements.txt', 'Pipfile', 'pyproject.toml',
        '*.pyc', '*.pyo', '*.so', '*.dll', '*.exe',
        '*.png', '*.jpg', '*.jpeg', '*.gif', '*.ico', '*.svg',
        '*.mp4', '*.mp3', '*.pdf', '*.doc', '*.docx',
        '*.log', '*.tmp', '*.cache', '*.db', '*.sqlite',
        '.DS_Store', 'Thumbs.db'
    }
    
    # Directorios a ignorar
    IGNORE_DIRS = {
        'node_modules', 'venv', 'env', '.venv', '__pycache__',
        '.git', '.vscode', '.idea', '.pytest_cache', '.mypy_cache',
        'dist', 'build', 'target', 'logs', 'tmp', 'temp',
        '.github', '.gitlab', '.circleci', 'assets', 'images'
    }
    
    def __init__(
        self,
        repo_name: str,
        repo_path: Path,
        repo_id: int,
        prefer_pro: bool = False,
        max_file_size_mb: int = 1,
        include_docs: bool = False
    ):
        """
        Inicializa servicio RAG optimizado.
        """
        self.repo_name = repo_name
        self.repo_path = repo_path
        self.repo_id = repo_id
        self.max_file_size_mb = max_file_size_mb
        self.include_docs = include_docs
        
        if include_docs:
            self.PRIORITY_EXTENSIONS.add('.md')
            self.PRIORITY_EXTENSIONS.add('.rst')
            self.PRIORITY_EXTENSIONS.add('.txt')
        
        # Inicializar componentes
        self.embedding = GeminiEmbedding()
        self.llm = GeminiLLM(prefer_pro=prefer_pro)
        self.cache = CacheService()
        
        # Obtener la dimensión real del embedding
        self.embedding_dimension = self.embedding.get_dimension()
        logger.info(f"Dimensión de embeddings detectada: {self.embedding_dimension}")
        
        # Índice FAISS específico para este repositorio
        safe_name = repo_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        index_path = Path(f"data/vectors/{safe_name}.index")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.vector_store = FAISSStore(
            dimension=self.embedding_dimension,
            index_path=index_path
        )
        
        # Control de rate limiting
        self._last_request_time = 0
        self._request_count = 0
        self._rate_limit_exceeded = False
        self._rate_limit_reset_time = 0
        self._cancelled = False
        self._daily_limit_reached = False
        
        # Estadísticas
        self.stats = {
            'total_files': 0,
            'valid_files': 0,
            'total_chunks': 0,
            'chunks_processed': 0,
            'api_calls': 0,
            'api_errors': 0,
            'rate_limit_hits': 0,
            'daily_limit_hit': False,
            'cache_hits': 0
        }
        
        logger.info("=" * 60)
        logger.info("RAGGeminiService OPTIMIZADO inicializado")
        logger.info(f"Repositorio: {repo_name} (ID: {repo_id})")
        logger.info(f"Dimensión embeddings: {self.embedding_dimension}")
        logger.info(f"Extensiones válidas: {len(self.PRIORITY_EXTENSIONS)}")
        logger.info(f"Fragmentos por archivo: {self.MAX_FRAGMENTS_PER_FILE}")
        logger.info(f"Tamaño fragmento: {self.MAX_CHUNK_SIZE} caracteres")
        logger.info(f"Lote de procesamiento: {self.BATCH_SIZE} fragmentos")
        logger.info("=" * 60)
    
    def _should_ignore_file(self, file_path: Path, file_name: str) -> bool:
        """Determina si un archivo debe ser ignorado."""
        if file_name in self.IGNORE_FILES:
            return True
        
        for pattern in self.IGNORE_FILES:
            if pattern.startswith('*') and file_name.endswith(pattern[1:]):
                return True
        
        for ignore_dir in self.IGNORE_DIRS:
            if ignore_dir in file_path.parts:
                return True
        
        return False
    
    def _is_valid_file(self, file) -> bool:
        """Verifica si un archivo es válido para indexación."""
        ext = file.extension.lower()
        
        if ext not in self.PRIORITY_EXTENSIONS:
            return False
        
        if self._should_ignore_file(file.path, file.name):
            return False
        
        if file.line_count > self.MAX_FILE_LINES:
            return False
        
        return True
    
    def _filter_valid_files(self, files: List) -> List:
        """Filtra archivos válidos."""
        valid_files = []
        
        for file in files:
            if self._is_valid_file(file):
                valid_files.append(file)
        
        self.stats['total_files'] = len(files)
        self.stats['valid_files'] = len(valid_files)
        
        logger.info(f"Filtrado: {len(valid_files)} válidos de {len(files)} totales")
        
        return valid_files
    
    def _chunk_code_optimized(self, code: str, file_name: str) -> List[str]:
        """Divide código en fragmentos optimizados."""
        lines = code.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('<!--'):
                continue
            
            line_size = len(line)
            
            if current_size + line_size > self.MAX_CHUNK_SIZE and current_chunk:
                chunk = '\n'.join(current_chunk)
                if len(chunk.strip()) > 50:
                    chunks.append(chunk)
                current_chunk = []
                current_size = 0
            
            current_chunk.append(line)
            current_size += line_size
            
            if len(chunks) >= self.MAX_FRAGMENTS_PER_FILE:
                break
        
        if current_chunk and len(chunks) < self.MAX_FRAGMENTS_PER_FILE:
            chunk = '\n'.join(current_chunk)
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
        
        logger.debug(f"Archivo {file_name}: {len(chunks)} fragmentos")
        return chunks
    
    def _wait_for_rate_limit(self) -> None:
        """Espera respetando rate limits."""
        if self._daily_limit_reached:
            logger.warning("Límite diario alcanzado")
            time.sleep(3600)
            return
        
        now = time.time()
        
        if self._rate_limit_exceeded:
            wait_time = max(0, self._rate_limit_reset_time - now)
            if wait_time > 0:
                logger.info(f"Rate limit, esperando {wait_time:.1f} segundos...")
                time.sleep(wait_time)
                self._rate_limit_exceeded = False
        
        min_interval = 1.0
        elapsed = now - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        self._last_request_time = time.time()
    
    def _generate_embedding_with_retry(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """Genera embedding con retry automático."""
        if self._daily_limit_reached:
            return None
        
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                
                vector = self.embedding.generate_embedding(text)
                self._request_count += 1
                self.stats['api_calls'] += 1
                self._rate_limit_exceeded = False
                
                if vector is None:
                    raise ValueError("Embedding generado es None")
                if len(vector) != self.embedding_dimension:
                    raise ValueError(f"Dimensión incorrecta: {len(vector)} != {self.embedding_dimension}")
                
                return vector
                
            except Exception as e:
                error_msg = str(e).lower()
                
                if '429' in error_msg or 'quota' in error_msg or 'rate limit' in error_msg:
                    self.stats['rate_limit_hits'] += 1
                    
                    if 'perday' in error_msg or 'daily' in error_msg:
                        self._daily_limit_reached = True
                        self.stats['daily_limit_hit'] = True
                        logger.error("LÍMITE DIARIO ALCANZADO")
                        return None
                    
                    self._rate_limit_exceeded = True
                    
                    match = re.search(r'retry in (\d+(?:\.\d+)?)', error_msg)
                    if match:
                        wait_time = float(match.group(1)) + 5
                    else:
                        wait_time = self.API_RETRY_BASE_DELAY * (attempt + 1)
                    
                    logger.warning(f"Rate limit, esperando {wait_time:.1f} segundos...")
                    time.sleep(wait_time)
                    self._rate_limit_reset_time = time.time() + wait_time
                    continue
                
                self.stats['api_errors'] += 1
                logger.error(f"Error generando embedding: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
        
        return None
    
    def index_repository(self, repository: Repository) -> bool:
        """Indexa repositorio completo."""
        start_time = time.time()
        self._cancelled = False
        
        try:
            logger.info(f"Iniciando indexación de {repository.name}")
            
            valid_files = self._filter_valid_files(repository.files)
            
            if not valid_files:
                logger.warning("No hay archivos válidos")
                return False
            
            all_chunks = []
            all_metadata = []
            
            logger.info("Generando fragmentos...")
            for file in valid_files:
                try:
                    file_path = self.repo_path / file.relative_path
                    if not file_path.exists():
                        continue
                    
                    if file_path.stat().st_size > self.max_file_size_mb * 1024 * 1024:
                        continue
                    
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    chunks = self._chunk_code_optimized(content, file.name)
                    
                    for i, chunk in enumerate(chunks):
                        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
                        
                        metadata = {
                            'repo_id': self.repo_id,
                            'repo': self.repo_name,
                            'file': file.relative_path,
                            'file_name': file.name,
                            'chunk_index': i,
                            'preview': preview
                        }
                        
                        all_chunks.append(chunk)
                        all_metadata.append(metadata)
                        
                except Exception as e:
                    logger.error(f"Error procesando {file.name}: {e}")
                    continue
            
            self.stats['total_chunks'] = len(all_chunks)
            logger.info(f"Fragmentos generados: {len(all_chunks)}")
            
            if not all_chunks:
                logger.warning("No se generaron fragmentos")
                return False
            
            logger.info("Procesando embeddings...")
            
            all_vectors = []
            all_ids = []
            all_metadatas = []
            
            for i in range(0, len(all_chunks), self.BATCH_SIZE):
                if self._cancelled or self._daily_limit_reached:
                    return False
                
                batch = all_chunks[i:i + self.BATCH_SIZE]
                batch_metadata = all_metadata[i:i + self.BATCH_SIZE]
                
                logger.info(f"Lote {i//self.BATCH_SIZE + 1}: {len(batch)} fragmentos")
                
                for idx, (chunk, metadata) in enumerate(zip(batch, batch_metadata)):
                    if self._daily_limit_reached:
                        return False
                    
                    vector = self._generate_embedding_with_retry(chunk)
                    
                    if vector is None:
                        continue
                    
                    if len(vector) != self.embedding_dimension:
                        continue
                    
                    chunk_id = f"{self.repo_id}:{metadata['file']}:{metadata['chunk_index']}"
                    
                    all_vectors.append(vector)
                    all_ids.append(chunk_id)
                    all_metadatas.append(metadata)
                    
                    self.stats['chunks_processed'] += 1
                    
                    if self.stats['chunks_processed'] % 20 == 0:
                        logger.info(f"Progreso: {self.stats['chunks_processed']}/{self.stats['total_chunks']}")
                
                if i + self.BATCH_SIZE < len(all_chunks):
                    logger.info(f"Pausa de {self.BATCH_DELAY} segundos...")
                    time.sleep(self.BATCH_DELAY)
            
            if all_vectors:
                self.vector_store.add_vectors(all_vectors, all_ids, all_metadatas)
                
                elapsed = time.time() - start_time
                logger.info("=" * 60)
                logger.info("INDEXACIÓN COMPLETADA")
                logger.info(f"Fragmentos: {self.stats['chunks_processed']}/{self.stats['total_chunks']}")
                logger.info(f"API Calls: {self.stats['api_calls']}")
                logger.info(f"Tiempo: {elapsed:.2f} segundos")
                logger.info("=" * 60)
                return True
            else:
                logger.error("No se generaron vectores")
                return False
                
        except Exception as e:
            logger.error(f"Error en indexación: {e}")
            return False
    
    def query(self, question: str, k: int = 5, include_sources: bool = True) -> Dict[str, Any]:
        """Realiza consulta RAG."""
        start_time = time.time()
        
        try:
            logger.info(f"Procesando consulta: {question[:100]}...")
            
            if len(question) > 500:
                question = question[:500]
            
            # Intentar generar embedding
            try:
                query_vector = self._generate_embedding_with_retry(question)
            except Exception as e:
                logger.error(f"Error generando embedding: {e}")
                return {
                    'answer': f"Error generando embedding: {str(e)}",
                    'sources': []
                }
            
            if query_vector is None:
                return {
                    'answer': "Error generando embedding. Intenta de nuevo.",
                    'sources': []
                }
            
            # Validar dimensión del embedding
            logger.debug(f"Embedding generado: {len(query_vector)} dimensiones")
            
            if len(query_vector) != self.embedding_dimension:
                logger.error(f"Embedding dimensión incorrecta: {len(query_vector)} != {self.embedding_dimension}")
                logger.error(f"Embedding muestra: {query_vector[:10] if query_vector else 'None'}")
                return {
                    'answer': f"Error: El embedding generado tiene dimensión {len(query_vector)} pero se esperaban {self.embedding_dimension}. Por favor, intenta de nuevo.",
                    'sources': []
                }
            
            # Buscar en FAISS
            results = self.vector_store.search(query_vector, k=min(k, 10))
            
            if not results:
                return {
                    'answer': "No encontré información relevante en el repositorio.",
                    'sources': []
                }
            
            context_parts = []
            sources_for_display = []
            
            for i, r in enumerate(results[:5]):
                metadata = r['metadata']
                preview = metadata.get('preview', '')[:300]
                
                context_parts.append(f"[{i+1}] {metadata['file']}\n{preview}\n")
                sources_for_display.append({
                    'file': metadata['file'],
                    'preview': preview,
                    'score': round(r.get('score', 0), 3)
                })
            
            context = "\n---\n".join(context_parts)
            prompt = self._build_prompt(question, context)
            answer = self.llm.generate(prompt)
            
            elapsed = time.time() - start_time
            logger.info(f"Consulta completada en {elapsed:.2f} segundos")
            
            return {
                'answer': answer,
                'sources': sources_for_display if include_sources else [],
                'model_used': self.llm.current_model,
                'elapsed_seconds': round(elapsed, 2)
            }
            
        except Exception as e:
            logger.error(f"Error en consulta: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'sources': []
            }
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Construye prompt optimizado."""
        return f"""Eres un experto en análisis de código. Responde basándote en el contexto.

                    CONTEXTO:
                    {context}

                    PREGUNTA: {question}

                    RESPUESTA:"""
    
    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas del servicio."""
        try:
            vector_stats = self.vector_store.get_stats()
        except Exception:
            vector_stats = {'total_vectors': 0}
        
        model_info = self.llm.get_model_info()
        
        return {
            'repository': {
                'name': self.repo_name,
                'id': self.repo_id
            },
            'vector_store': vector_stats,
            'llm': model_info,
            'processing_stats': self.stats,
            'daily_limit_reached': self._daily_limit_reached,
            'embedding_dimension': self.embedding_dimension
        }