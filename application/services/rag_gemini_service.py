"""
Servicio RAG unificado y optimizado con soporte de caché.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import hashlib

from infrastructure.embeddings.gemini_embedding import GeminiEmbedding
from infrastructure.vector_db.faiss_store import FAISSStore
from infrastructure.llm.gemini_llm import GeminiLLM
from domain.models.repository import Repository
from application.services.cache_service import CacheService
from infrastructure.database.mysql_repository import MySQLRepository

logger = logging.getLogger(__name__)


class RAGService:
    """
    Servicio RAG unificado y optimizado con soporte de caché.
    
    Responsabilidades:
    - Indexar repositorios (embeddings + FAISS)
    - Responder consultas usando contexto recuperado
    - Gestionar caché de archivos y consultas
    """
    
    def __init__(
        self,
        repo_name: str,
        repo_path: Path,
        repo_id: int,
        prefer_pro: bool = False
    ):
        """
        Inicializa servicio RAG.
        
        Args:
            repo_name: Nombre del repositorio
            repo_path: Ruta física del repositorio
            repo_id: ID en base de datos
            prefer_pro: Preferir modelos Pro de Gemini
        """
        self.repo_name = repo_name
        self.repo_path = repo_path
        self.repo_id = repo_id
        self.db = MySQLRepository()
        
        # Inicializar componentes
        self.embedding = GeminiEmbedding()
        self.llm = GeminiLLM(prefer_pro=prefer_pro)
        self.cache = CacheService()
        
        # Caché de consultas frecuentes
        self.query_cache = {}
        self.query_cache_size = 100
        self.query_hits = 0
        self.query_misses = 0
        
        # Índice FAISS específico para este repo
        safe_name = repo_name.replace(' ', '_').replace('/', '_')
        index_path = Path(f"data/vectors/{safe_name}.index")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.vector_store = FAISSStore(
            dimension=self.embedding.get_dimension(),
            index_path=index_path
        )
        
        model_info = self.llm.get_model_info()
        
        logger.info(f"RAGService inicializado para {repo_name}")
        logger.info(f"  ID: {repo_id}")
        logger.info(f"  Ruta: {repo_path}")
        logger.info(f"  Modelo: {model_info['current_model']}")
        logger.info(f"  Índice: {index_path}")
    
    def index_repository(self, repository: Repository) -> bool:
        """
        Indexa repositorio completo en FAISS usando caché.
        
        Args:
            repository: Repositorio a indexar
            
        Returns:
            True si éxito
        """
        start_time = time.time()
        
        try:
            logger.info(f"Iniciando indexación de {repository.name}")
            
            all_vectors = []
            all_ids = []
            all_metadatas = []
            
            total_files = len(repository.files)
            files_processed = 0
            
            for idx, file in enumerate(repository.files, 1):
                logger.info(f"Procesando [{idx}/{total_files}]: {file.name}")
                
                # Intentar obtener del caché primero
                cached_content = self.cache.get_text(self.repo_id, file.relative_path)
                
                if cached_content:
                    content = cached_content
                    logger.debug(f"Archivo obtenido de caché: {file.relative_path}")
                else:
                    # Leer de disco
                    file_path = self.repo_path / file.relative_path
                    
                    if not file_path.exists():
                        logger.warning(f"Archivo no encontrado: {file_path}")
                        continue
                    
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        
                        # Guardar en caché para futuro
                        self.cache.put_text(self.repo_id, file.relative_path, content)
                        logger.debug(f"Archivo guardado en caché: {file.relative_path}")
                        
                    except Exception as e:
                        logger.error(f"Error leyendo archivo {file_path}: {e}")
                        continue
                
                if not content.strip():
                    logger.warning(f"Archivo vacío: {file.name}")
                    continue
                
                # Dividir en fragmentos
                chunks = self._chunk_code(content)
                
                # Procesar cada fragmento
                for i, chunk in enumerate(chunks):
                    # Generar embedding
                    try:
                        vector = self.embedding.generate_embedding(chunk)
                    except Exception as e:
                        logger.error(f"Error generando embedding: {e}")
                        continue
                    
                    # Crear ID único
                    chunk_id = f"{self.repo_id}:{file.relative_path}:{i}"
                    
                    # Metadatos
                    preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
                    
                    metadata = {
                        'repo_id': self.repo_id,
                        'repo': repository.name,
                        'file': file.relative_path,
                        'file_name': file.name,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'preview': preview,
                        'functions': len(file.functions),
                        'classes': len(file.classes)
                    }
                    
                    all_vectors.append(vector)
                    all_ids.append(chunk_id)
                    all_metadatas.append(metadata)
                
                files_processed += 1
            
            # Guardar en FAISS
            if all_vectors:
                self.vector_store.add_vectors(all_vectors, all_ids, all_metadatas)
                
                elapsed = time.time() - start_time
                logger.info(f"Indexación completada en {elapsed:.2f} segundos:")
                logger.info(f"  Archivos procesados: {files_processed}/{total_files}")
                logger.info(f"  Fragmentos generados: {len(all_vectors)}")
                logger.info(f"  Promedio: {len(all_vectors)/files_processed:.1f} frag/archivo")
                
                return True
            else:
                logger.error("No se generaron vectores para indexar")
                return False
                
        except Exception as e:
            logger.error(f"Error en indexación: {e}")
            return False
    
    def _chunk_code(self, code: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Divide código en fragmentos para embedding.
        
        Args:
            code: Código fuente
            chunk_size: Tamaño de fragmento
            overlap: Superposición entre fragmentos
            
        Returns:
            Lista de fragmentos
        """
        chunks = []
        start = 0
        code_length = len(code)
        
        while start < code_length:
            end = min(start + chunk_size, code_length)
            
            # Buscar fin de línea
            if end < code_length:
                next_newline = code.find('\n', end)
                if next_newline != -1 and next_newline - end < 100:
                    end = next_newline + 1
            
            chunk = code[start:end]
            chunks.append(chunk)
            
            # Avanzar con superposición
            start = end - overlap
        
        return chunks
    
    def query(
        self,
        question: str,
        k: int = 5,
        include_sources: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Realiza consulta RAG con caché de consultas.
        
        Args:
            question: Pregunta del usuario
            k: Número de fragmentos a recuperar
            include_sources: Incluir fuentes en respuesta
            use_cache: Usar caché de consultas
            
        Returns:
            Respuesta con fuentes
        """
        start_time = time.time()
        query_hash = hashlib.md5(question.encode()).hexdigest()
        
        # Verificar caché de consultas
        if use_cache and query_hash in self.query_cache:
            self.query_hits += 1
            logger.info(f"Cache hit para consulta (total hits: {self.query_hits})")
            return self.query_cache[query_hash]
        
        self.query_misses += 1
        
        try:
            logger.info(f"Procesando consulta: {question[:100]}...")
            
            # 1. Generar embedding de la pregunta
            query_vector = self.embedding.generate_embedding(question)
            
            # 2. Buscar en FAISS
            results = self.vector_store.search(query_vector, k=k)
            
            if not results:
                response = {
                    'answer': "No encontré información relevante en el repositorio.",
                    'sources': []
                }
                
                # Guardar en caché
                if use_cache:
                    self._cache_query(query_hash, response)
                
                return response
            
            # 3. Construir contexto
            context_parts = []
            sources_for_display = []
            
            for i, r in enumerate(results, 1):
                metadata = r['metadata']
                
                # Intentar obtener contenido completo del caché
                full_content = self.cache.get_text(
                    metadata['repo_id'],
                    metadata['file']
                )
                
                if full_content:
                    preview = full_content[:500] + "..." if len(full_content) > 500 else full_content
                else:
                    preview = metadata.get('preview', '')
                
                context_parts.append(
                    f"[Fuente {i} - Archivo: {metadata['file']}]\n"
                    f"{preview}\n"
                )
                
                sources_for_display.append({
                    'file': metadata['file'],
                    'preview': preview[:200] + "..." if len(preview) > 200 else preview,
                    'score': round(r.get('score', 0), 3)
                })
            
            context = "\n---\n".join(context_parts)
            
            # 4. Generar respuesta
            prompt = self._build_prompt(question, context)
            answer = self.llm.generate(prompt)
            
            elapsed = time.time() - start_time
            
            response = {
                'answer': answer,
                'sources': sources_for_display if include_sources else [],
                'model_used': self.llm.current_model,
                'elapsed_seconds': round(elapsed, 2),
                'fragments_used': len(results),
                'cache_stats': {
                    'hits': self.query_hits,
                    'misses': self.query_misses,
                    'hit_rate': round(self.query_hits / (self.query_hits + self.query_misses) * 100, 2)
                }
            }
            
            # Guardar en caché
            if use_cache:
                self._cache_query(query_hash, response)
            
            logger.info(f"Consulta procesada en {elapsed:.2f} segundos")
            logger.info(f"  Fragmentos usados: {len(results)}")
            logger.info(f"  Cache hit rate: {response['cache_stats']['hit_rate']}%")
            
            return response
            
        except Exception as e:
            logger.error(f"Error en consulta: {e}")
            return {
                'answer': f"Error procesando consulta: {str(e)}",
                'sources': []
            }
    
    def _cache_query(self, query_hash: str, response: Dict[str, Any]) -> None:
        """
        Guarda consulta en caché LRU.
        
        Args:
            query_hash: Hash de la consulta
            response: Respuesta a cachear
        """
        # Si el caché está lleno, eliminar el más antiguo
        if len(self.query_cache) >= self.query_cache_size:
            oldest_key = min(self.query_cache.keys(), 
                           key=lambda k: self.query_cache[k].get('timestamp', 0))
            del self.query_cache[oldest_key]
        
        # Guardar con timestamp
        response['timestamp'] = time.time()
        self.query_cache[query_hash] = response
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Construye prompt optimizado para Gemini."""
        return f"""Eres un experto en análisis de código. Responde basándote ÚNICAMENTE en el contexto proporcionado.

CONTEXTO DEL REPOSITORIO:
{context}

PREGUNTA: {question}

INSTRUCCIONES:
- Responde SOLO con información del contexto
- Si la información no está en el contexto, indícalo claramente
- Sé técnico y preciso
- Usa formato de código con ``` cuando sea necesario
- Si encuentras errores potenciales, menciónalos constructivamente

RESPUESTA:"""
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Estadísticas completas del servicio.
        
        Returns:
            Diccionario con estadísticas
        """
        vector_stats = self.vector_store.get_stats()
        model_info = self.llm.get_model_info()
        cache_stats = self.cache.get_stats()
        
        return {
            'repository': {
                'name': self.repo_name,
                'id': self.repo_id,
                'path': str(self.repo_path)
            },
            'vector_store': vector_stats,
            'llm': model_info,
            'cache': cache_stats,
            'query_cache': {
                'size': len(self.query_cache),
                'max_size': self.query_cache_size,
                'hits': self.query_hits,
                'misses': self.query_misses,
                'hit_rate': round(self.query_hits / (self.query_hits + self.query_misses) * 100, 2) if (self.query_hits + self.query_misses) > 0 else 0
            },
            'embedding_dimension': self.embedding.get_dimension()
        }