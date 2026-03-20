"""
Servicio RAG con Gemini y FAISS.
Versión optimizada con control de rate limiting y manejo robusto de errores.
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
    
    # Constantes optimizadas para free tier de Gemini API
    MAX_CHUNK_SIZE = 500               # Caracteres por fragmento
    CHUNK_OVERLAP = 100                # Superposición entre fragmentos
    MAX_FRAGMENTS_PER_FILE = 10        # Máximo fragmentos por archivo
    MAX_FILE_SIZE_MB = 1               # Tamaño máximo de archivo en MB
    MAX_FILE_LINES = 2000              # Máximo líneas por archivo
    API_TIMEOUT_SECONDS = 30           # Timeout para API calls
    BATCH_SIZE = 20                    # Fragmentos por lote
    BATCH_DELAY = 2                    # Segundos entre lotes
    API_RETRY_BASE_DELAY = 30          # Segundos base para retry en error 429
    
    # Extensiones prioritarias (código fuente)
    PRIORITY_EXTENSIONS = {
        '.py', '.pyx',                  # Python
        '.js', '.jsx', '.mjs', '.cjs',  # JavaScript
        '.ts', '.tsx',                  # TypeScript
        '.html', '.htm',                # HTML
        '.css', '.scss', '.sass',       # CSS
        '.json', '.yaml', '.yml',       # Configuración
        '.sql',                         # SQL
        '.sh', '.bash', '.zsh',         # Shell
        '.go', '.rs', '.java', '.cpp', '.c', '.h', '.hpp',  # Compilados
        '.rb', '.php', '.vue'           # Otros
    }
    
    # Archivos a ignorar (por nombre exacto o patrón)
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
        
        Args:
            repo_name: Nombre del repositorio
            repo_path: Ruta física del repositorio
            repo_id: ID en base de datos
            prefer_pro: Preferir modelos Pro de Gemini
            max_file_size_mb: Tamaño máximo de archivo en MB
            include_docs: Incluir archivos de documentación
        """
        self.repo_name = repo_name
        self.repo_path = repo_path
        self.repo_id = repo_id
        self.max_file_size_mb = max_file_size_mb
        self.include_docs = include_docs
        
        # Si se incluye documentación, agregar extensiones
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
        
        # Estadísticas
        self.stats = {
            'total_files': 0,
            'valid_files': 0,
            'total_chunks': 0,
            'chunks_processed': 0,
            'api_calls': 0,
            'api_errors': 0,
            'rate_limit_hits': 0,
            'cache_hits': 0
        }
        
        logger.info("=" * 60)
        logger.info("RAGGeminiService OPTIMIZADO inicializado")
        logger.info(f"Repositorio: {repo_name} (ID: {repo_id})")
        logger.info(f"Ruta: {repo_path}")
        logger.info(f"Dimensión embeddings: {self.embedding_dimension}")
        logger.info(f"Extensiones válidas: {len(self.PRIORITY_EXTENSIONS)}")
        logger.info(f"Fragmentos por archivo: {self.MAX_FRAGMENTS_PER_FILE}")
        logger.info(f"Tamaño fragmento: {self.MAX_CHUNK_SIZE} caracteres")
        logger.info(f"Tamaño máximo archivo: {max_file_size_mb} MB")
        logger.info(f"Lote de procesamiento: {self.BATCH_SIZE} fragmentos")
        logger.info(f"Delay entre lotes: {self.BATCH_DELAY} segundos")
        logger.info("=" * 60)
    
    def _should_ignore_file(self, file_path: Path, file_name: str) -> bool:
        """
        Determina si un archivo debe ser ignorado.
        
        Args:
            file_path: Ruta completa del archivo
            file_name: Nombre del archivo
            
        Returns:
            True si debe ser ignorado
        """
        # Verificar nombre exacto
        if file_name in self.IGNORE_FILES:
            return True
        
        # Verificar por patrón (*.ext)
        for pattern in self.IGNORE_FILES:
            if pattern.startswith('*') and file_name.endswith(pattern[1:]):
                return True
        
        # Verificar directorios a ignorar
        for ignore_dir in self.IGNORE_DIRS:
            if ignore_dir in file_path.parts:
                return True
        
        return False
    
    def _is_valid_file(self, file) -> bool:
        """
        Verifica si un archivo es válido para indexación.
        
        Args:
            file: Archivo a verificar
            
        Returns:
            True si es válido
        """
        ext = file.extension.lower()
        
        # Verificar extensión
        if ext not in self.PRIORITY_EXTENSIONS:
            return False
        
        # Verificar nombre de archivo
        if self._should_ignore_file(file.path, file.name):
            return False
        
        # Verificar tamaño por líneas
        if file.line_count > self.MAX_FILE_LINES:
            logger.debug(f"Archivo ignorado por líneas: {file.name} ({file.line_count})")
            return False
        
        return True
    
    def _filter_valid_files(self, files: List) -> List:
        """
        Filtra archivos válidos para indexación.
        
        Args:
            files: Lista de archivos
            
        Returns:
            Lista de archivos válidos
        """
        valid_files = []
        
        for file in files:
            if self._is_valid_file(file):
                valid_files.append(file)
        
        self.stats['total_files'] = len(files)
        self.stats['valid_files'] = len(valid_files)
        
        logger.info(f"Filtrado completado: {len(valid_files)} válidos de {len(files)} totales")
        
        return valid_files
    
    def _chunk_code_optimized(self, code: str, file_name: str) -> List[str]:
        """
        Divide código en fragmentos optimizados.
        
        Args:
            code: Código fuente
            file_name: Nombre del archivo (para logs)
            
        Returns:
            Lista de fragmentos
        """
        lines = code.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            # Ignorar líneas vacías o solo comentarios
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('<!--'):
                continue
            
            line_size = len(line)
            
            # Si agregar esta línea excede el tamaño, guardar fragmento actual
            if current_size + line_size > self.MAX_CHUNK_SIZE and current_chunk:
                chunk = '\n'.join(current_chunk)
                if len(chunk.strip()) > 50:
                    chunks.append(chunk)
                current_chunk = []
                current_size = 0
            
            current_chunk.append(line)
            current_size += line_size
            
            # Límite de fragmentos por archivo
            if len(chunks) >= self.MAX_FRAGMENTS_PER_FILE:
                break
        
        # Último fragmento
        if current_chunk and len(chunks) < self.MAX_FRAGMENTS_PER_FILE:
            chunk = '\n'.join(current_chunk)
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
        
        logger.debug(f"Archivo {file_name}: {len(chunks)} fragmentos")
        return chunks
    
    def _wait_for_rate_limit(self) -> None:
        """
        Espera respetando rate limits de la API.
        """
        now = time.time()
        
        # Si estamos en rate limit, esperar
        if self._rate_limit_exceeded:
            wait_time = max(0, self._rate_limit_reset_time - now)
            if wait_time > 0:
                logger.info(f"Rate limit activo, esperando {wait_time:.1f} segundos...")
                time.sleep(wait_time)
                self._rate_limit_exceeded = False
        
        # Limitar solicitudes por segundo (máximo 2 por segundo)
        min_interval = 0.5
        elapsed = now - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        self._last_request_time = time.time()
    
    def _generate_embedding_with_retry(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """
        Genera embedding con retry automático para rate limits.
        
        Args:
            text: Texto para embedding
            max_retries: Número máximo de reintentos
            
        Returns:
            Vector de embeddings o None si falla
        """
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                
                vector = self.embedding.generate_embedding(text)
                self._request_count += 1
                self.stats['api_calls'] += 1
                self._rate_limit_exceeded = False
                
                # Validar vector
                if vector is None:
                    raise ValueError("Embedding generado es None")
                if len(vector) != self.embedding_dimension:
                    raise ValueError(f"Dimensión incorrecta: {len(vector)} != {self.embedding_dimension}")
                
                return vector
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Detectar rate limit (error 429)
                if '429' in error_msg or 'quota' in error_msg or 'rate limit' in error_msg:
                    self.stats['rate_limit_hits'] += 1
                    self._rate_limit_exceeded = True
                    
                    # Extraer tiempo de espera si está disponible
                    import re
                    match = re.search(r'retry in (\d+(?:\.\d+)?)', error_msg)
                    if match:
                        wait_time = float(match.group(1)) + 5
                    else:
                        wait_time = self.API_RETRY_BASE_DELAY * (attempt + 1)
                    
                    logger.warning(f"Rate limit detectado, esperando {wait_time:.1f} segundos...")
                    time.sleep(wait_time)
                    self._rate_limit_reset_time = time.time() + wait_time
                    continue
                
                # Otros errores
                self.stats['api_errors'] += 1
                logger.error(f"Error generando embedding (intento {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Reintentando en {wait_time} segundos...")
                    time.sleep(wait_time)
                    continue
                
                return None
        
        return None
    
    def index_repository(self, repository: Repository) -> bool:
        """
        Indexa repositorio completo en FAISS.
        
        Args:
            repository: Repositorio a indexar
            
        Returns:
            True si éxito
        """
        start_time = time.time()
        self._cancelled = False
        self._request_count = 0
        
        try:
            logger.info(f"Iniciando indexación de {repository.name}")
            
            # 1. Filtrar archivos válidos
            valid_files = self._filter_valid_files(repository.files)
            
            if not valid_files:
                logger.warning("No hay archivos válidos para indexar")
                return False
            
            # 2. Generar fragmentos (sin API)
            all_chunks = []
            all_metadata = []
            
            logger.info("Generando fragmentos de código...")
            for file in valid_files:
                try:
                    file_path = self.repo_path / file.relative_path
                    if not file_path.exists():
                        logger.debug(f"Archivo no encontrado: {file_path}")
                        continue
                    
                    # Verificar tamaño físico
                    file_size_bytes = file_path.stat().st_size
                    if file_size_bytes > self.max_file_size_mb * 1024 * 1024:
                        logger.debug(f"Archivo ignorado por tamaño: {file.name} ({file_size_bytes / 1024 / 1024:.2f} MB)")
                        continue
                    
                    # Leer contenido
                    try:
                        content = file_path.read_text(encoding='utf-8')
                    except UnicodeDecodeError:
                        try:
                            content = file_path.read_text(encoding='latin-1')
                        except Exception:
                            logger.debug(f"No se pudo leer archivo: {file.name}")
                            continue
                    
                    if not content.strip():
                        logger.debug(f"Archivo vacío: {file.name}")
                        continue
                    
                    # Generar fragmentos
                    chunks = self._chunk_code_optimized(content, file.name)
                    
                    for i, chunk in enumerate(chunks):
                        chunk_id = f"{self.repo_id}:{file.relative_path}:{i}"
                        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
                        
                        metadata = {
                            'repo_id': self.repo_id,
                            'repo': self.repo_name,
                            'file': file.relative_path,
                            'file_name': file.name,
                            'chunk_index': i,
                            'preview': preview,
                            'language': file.extension[1:] if file.extension else 'unknown'
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
            
            # 3. Procesar embeddings en lotes (con API)
            logger.info("Procesando embeddings en lotes...")
            
            all_vectors = []
            all_ids = []
            all_metadatas = []
            
            for i in range(0, len(all_chunks), self.BATCH_SIZE):
                if self._cancelled:
                    logger.info("Indexación cancelada")
                    return False
                
                batch = all_chunks[i:i + self.BATCH_SIZE]
                batch_metadata = all_metadata[i:i + self.BATCH_SIZE]
                
                logger.info(f"Lote {i//self.BATCH_SIZE + 1}: {len(batch)} fragmentos")
                
                for idx, (chunk, metadata) in enumerate(zip(batch, batch_metadata)):
                    if self._cancelled:
                        return False
                    
                    # Generar embedding con retry automático
                    vector = self._generate_embedding_with_retry(chunk)
                    
                    if vector is None:
                        logger.error(f"Error generando embedding para fragmento {i + idx}")
                        continue
                    
                    # Validar vector antes de agregar
                    if len(vector) != self.embedding_dimension:
                        logger.error(f"Vector dimensión incorrecta: {len(vector)} != {self.embedding_dimension}")
                        continue
                    
                    chunk_id = f"{self.repo_id}:{metadata['file']}:{metadata['chunk_index']}"
                    
                    all_vectors.append(vector)
                    all_ids.append(chunk_id)
                    all_metadatas.append(metadata)
                    
                    self.stats['chunks_processed'] += 1
                    
                    # Mostrar progreso cada 20 fragmentos
                    if self.stats['chunks_processed'] % 20 == 0:
                        logger.info(f"Progreso: {self.stats['chunks_processed']}/{self.stats['total_chunks']} fragmentos")
                
                # Pausa entre lotes
                if i + self.BATCH_SIZE < len(all_chunks):
                    logger.info(f"Pausa de {self.BATCH_DELAY} segundos entre lotes...")
                    time.sleep(self.BATCH_DELAY)
            
            # 4. Verificar que hay vectores antes de guardar
            if not all_vectors:
                logger.error("No se generaron vectores válidos - todos los embeddings fallaron")
                return False
            
            logger.info(f"Vectores generados: {len(all_vectors)}")
            
            # Validar todos los vectores tienen la dimensión correcta
            valid_vectors = []
            valid_ids = []
            valid_metadatas = []
            
            for vec, vid, meta in zip(all_vectors, all_ids, all_metadatas):
                if vec is not None and len(vec) == self.embedding_dimension:
                    valid_vectors.append(vec)
                    valid_ids.append(vid)
                    valid_metadatas.append(meta)
                else:
                    logger.warning(f"Vector inválido descartado: dimensión {len(vec) if vec else 0}")
            
            if not valid_vectors:
                logger.error("No hay vectores con dimensión correcta")
                return False
            
            logger.info(f"Vectores válidos para FAISS: {len(valid_vectors)}")
            
            # 5. Guardar en FAISS
            try:
                self.vector_store.add_vectors(valid_vectors, valid_ids, valid_metadatas)
            except Exception as e:
                logger.error(f"Error guardando en FAISS: {e}")
                logger.error(traceback.format_exc())
                return False
            
            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info("INDEXACIÓN COMPLETADA")
            logger.info(f"Archivos válidos: {self.stats['valid_files']}/{self.stats['total_files']}")
            logger.info(f"Fragmentos procesados: {self.stats['chunks_processed']}/{self.stats['total_chunks']}")
            logger.info(f"Vectores guardados: {len(valid_vectors)}")
            logger.info(f"Solicitudes API: {self.stats['api_calls']}")
            logger.info(f"Rate limit hits: {self.stats['rate_limit_hits']}")
            logger.info(f"Errores API: {self.stats['api_errors']}")
            logger.info(f"Tiempo total: {elapsed:.2f} segundos")
            logger.info("=" * 60)
            
            return True
                
        except Exception as e:
            logger.error(f"Error en indexación: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def cancel_indexing(self) -> None:
        """Cancela el proceso de indexación en curso."""
        self._cancelled = True
        logger.info("Indexación cancelada")
    
    def query(
        self,
        question: str,
        k: int = 5,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Realiza consulta RAG.
        
        Args:
            question: Pregunta del usuario
            k: Número de fragmentos a recuperar
            include_sources: Incluir fuentes en respuesta
            
        Returns:
            Diccionario con respuesta y fuentes
        """
        start_time = time.time()
        
        try:
            logger.info(f"Procesando consulta: {question[:100]}...")
            
            # Limitar longitud de pregunta
            if len(question) > 500:
                question = question[:500]
                logger.debug("Pregunta truncada a 500 caracteres")
            
            # Generar embedding de la pregunta
            query_vector = self._generate_embedding_with_retry(question)
            if query_vector is None:
                return {
                    'answer': "Error generando embedding para la consulta. Intenta de nuevo.",
                    'sources': []
                }
            
            # Buscar en FAISS
            results = self.vector_store.search(query_vector, k=min(k, 10))
            
            if not results:
                return {
                    'answer': "No encontré información relevante en el repositorio.",
                    'sources': []
                }
            
            # Construir contexto
            context_parts = []
            sources_for_display = []
            
            for i, r in enumerate(results[:5]):
                metadata = r['metadata']
                preview = metadata.get('preview', '')[:300]
                
                context_parts.append(
                    f"[{i+1}] Archivo: {metadata['file']}\n"
                    f"{preview}\n"
                )
                sources_for_display.append({
                    'file': metadata['file'],
                    'preview': preview,
                    'score': round(r.get('score', 0), 3)
                })
            
            context = "\n---\n".join(context_parts)
            
            # Generar respuesta
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
                'answer': f"Error procesando consulta: {str(e)}",
                'sources': []
            }
    
    def _build_prompt(self, question: str, context: str) -> str:
        """
        Construye prompt optimizado para Gemini.
        
        Args:
            question: Pregunta del usuario
            context: Contexto recuperado
            
        Returns:
            Prompt completo
        """
        return f"""Eres un experto en análisis de código. Responde basándote en el contexto proporcionado.

CONTEXTO DEL REPOSITORIO:
{context}

PREGUNTA: {question}

INSTRUCCIONES:
- Responde SOLO con información del contexto
- Si la información no está en el contexto, indícalo claramente
- Sé breve y conciso
- Usa formato de código con ``` cuando sea necesario

RESPUESTA:"""
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Estadísticas completas del servicio.
        
        Returns:
            Diccionario con estadísticas
        """
        try:
            vector_stats = self.vector_store.get_stats()
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas vector store: {e}")
            vector_stats = {'total_vectors': 0}
        
        try:
            model_info = self.llm.get_model_info()
        except Exception as e:
            logger.error(f"Error obteniendo información del modelo: {e}")
            model_info = {'current_model': 'unknown', 'model_type': 'unknown'}
        
        return {
            'repository': {
                'name': self.repo_name,
                'id': self.repo_id,
                'path': str(self.repo_path)
            },
            'vector_store': vector_stats,
            'llm': model_info,
            'processing_stats': self.stats,
            'embedding_dimension': self.embedding_dimension,
            'limits': {
                'max_chunk_size': self.MAX_CHUNK_SIZE,
                'max_fragments_per_file': self.MAX_FRAGMENTS_PER_FILE,
                'max_file_size_mb': self.max_file_size_mb,
                'batch_size': self.BATCH_SIZE,
                'batch_delay': self.BATCH_DELAY
            },
            'filters': {
                'priority_extensions': list(self.PRIORITY_EXTENSIONS),
                'ignore_files_count': len(self.IGNORE_FILES),
                'ignore_dirs_count': len(self.IGNORE_DIRS)
            }
        }