"""
Servicio RAG con Gemini y FAISS.
Optimizado para eficiencia y escalabilidad.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from infrastructure.embeddings.gemini_embedding import GeminiEmbedding
from infrastructure.vector_db.faiss_store import FAISSStore
from infrastructure.llm_clients.gemini_llm import GeminiLLM
from domain.models.repository import Repository
from application.services.cache_service import CacheService

logger = logging.getLogger(__name__)


class RAGService:
    """
    Servicio RAG optimizado.
    
    Responsabilidades:
    - Indexar repositorios (embeddings + FAISS)
    - Responder consultas usando contexto recuperado
    - Gestionar caché de fragmentos y embeddings
    """
    
    # Constantes optimizadas
    MAX_CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    MAX_FRAGMENTS_PER_FILE = 10
    BATCH_SIZE = 20
    BATCH_DELAY = 2
    
    def __init__(
        self,
        repo_name: str,
        repo_path: Path,
        repo_id: int,
        model_name: str = "gemini-2.5-flash",
        max_file_size_mb: int = 1,
        include_docs: bool = False
    ):
        """
        Inicializa servicio RAG.
        
        Args:
            repo_name: Nombre del repositorio
            repo_path: Ruta física
            repo_id: ID en base de datos
            model_name: Modelo de Gemini a usar
            max_file_size_mb: Tamaño máximo de archivo
            include_docs: Incluir documentación
        """
        self.repo_name = repo_name
        self.repo_path = repo_path
        self.repo_id = repo_id
        self.model_name = model_name
        self.max_file_size_mb = max_file_size_mb
        self.include_docs = include_docs
        
        # Inicializar componentes
        self.embedding = GeminiEmbedding()
        self.llm = GeminiLLM(model=model_name, auto_fallback=False)
        self.cache = CacheService()
        
        # Dimensión de embeddings
        self.embedding_dimension = self.embedding.get_dimension()
        
        # Índice FAISS específico para este repositorio
        safe_name = repo_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        index_path = Path(f"data/vectors/{safe_name}.index")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.vector_store = FAISSStore(
            dimension=self.embedding_dimension,
            index_path=index_path
        )
        
        # Estadísticas
        self.stats = {
            "total_chunks": 0,
            "chunks_processed": 0,
            "api_calls": 0,
            "cache_hits": 0,
        }
        
        logger.info(f"RAGService inicializado | Repo: {repo_name} | Modelo: {model_name}")
    
    # ==================== MÉTODOS PRIVADOS ====================
    
    def _chunk_code(self, text: str) -> List[str]:
        """
        Divide código en fragmentos optimizados.
        
        Args:
            text: Código fuente
            
        Returns:
            Lista de fragmentos
        """
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len and len(chunks) < self.MAX_FRAGMENTS_PER_FILE:
            end = start + self.MAX_CHUNK_SIZE
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk)
            
            start = end - self.CHUNK_OVERLAP
        
        return chunks
    
    def _process_file(self, file) -> List[str]:
        """
        Procesa un archivo y genera fragmentos.
        
        Args:
            file: Archivo a procesar
            
        Returns:
            Lista de IDs de fragmentos generados
        """
        chunk_ids = []
        
        try:
            file_path = self.repo_path / file.relative_path
            if not file_path.exists():
                return chunk_ids
            
            # Verificar tamaño
            if file_path.stat().st_size > self.max_file_size_mb * 1024 * 1024:
                return chunk_ids
            
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            chunks = self._chunk_code(content)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{self.repo_id}:{file.relative_path}:{i}"
                self.cache.put_chunk(chunk_id, chunk)
                chunk_ids.append(chunk_id)
                
        except Exception as e:
            logger.error(f"Error procesando {file.relative_path}: {e}")
        
        return chunk_ids
    
    def _generate_embeddings_batch(self, chunk_ids: List[str]) -> tuple:
        """
        Genera embeddings para un lote de fragmentos.
        
        Args:
            chunk_ids: Lista de IDs de fragmentos
            
        Returns:
            Tupla (vectors, ids) con los vectores generados
        """
        batch_texts = []
        valid_ids = []
        
        # Preparar lote
        for chunk_id in chunk_ids:
            # Verificar caché de embeddings
            cached_vector = self.cache.get_embedding(chunk_id)
            if cached_vector:
                self.stats["cache_hits"] += 1
                return [cached_vector], [chunk_id]
            
            content = self.cache.get_chunk(chunk_id)
            if content:
                batch_texts.append(content)
                valid_ids.append(chunk_id)
        
        if not batch_texts:
            return [], []
        
        # Generar embeddings en batch
        try:
            vectors = self.embedding.generate_embeddings_batch(batch_texts)
            self.stats["api_calls"] += 1
            
            valid_vectors = []
            valid_chunk_ids = []
            
            for chunk_id, vector in zip(valid_ids, vectors):
                if vector and len(vector) == self.embedding_dimension:
                    self.cache.put_embedding(chunk_id, vector)
                    valid_vectors.append(vector)
                    valid_chunk_ids.append(chunk_id)
                    self.stats["chunks_processed"] += 1
            
            return valid_vectors, valid_chunk_ids
            
        except Exception as e:
            logger.error(f"Error generando embeddings batch: {e}")
            return [], []
    
    # ==================== MÉTODOS PÚBLICOS ====================
    
    def index_repository(self, repository: Repository) -> bool:
        """
        Indexa repositorio completo.
        
        Args:
            repository: Repositorio a indexar
            
        Returns:
            True si éxito
        """
        start_time = time.time()
        
        try:
            # 1. Generar fragmentos
            all_chunk_ids = []
            for file in repository.files:
                chunk_ids = self._process_file(file)
                all_chunk_ids.extend(chunk_ids)
            
            self.stats["total_chunks"] = len(all_chunk_ids)
            
            if not all_chunk_ids:
                logger.warning("No se generaron fragmentos")
                return False
            
            logger.info(f"Fragmentos generados: {len(all_chunk_ids)}")
            
            # 2. Generar embeddings por lotes
            all_vectors = []
            final_ids = []
            
            for i in range(0, len(all_chunk_ids), self.BATCH_SIZE):
                batch_ids = all_chunk_ids[i:i + self.BATCH_SIZE]
                
                vectors, valid_ids = self._generate_embeddings_batch(batch_ids)
                
                if vectors:
                    all_vectors.extend(vectors)
                    final_ids.extend(valid_ids)
                
                # Pausa entre lotes
                if i + self.BATCH_SIZE < len(all_chunk_ids):
                    time.sleep(self.BATCH_DELAY)
            
            # 3. Guardar en FAISS
            if all_vectors:
                self.vector_store.add_vectors(all_vectors, final_ids)
                elapsed = time.time() - start_time
                logger.info(f"Indexación completada: {self.stats['chunks_processed']}/{self.stats['total_chunks']} chunks | "
                           f"API: {self.stats['api_calls']} | Tiempo: {elapsed:.2f}s")
                return True
            
            logger.warning("No se generaron vectores")
            return False
            
        except Exception as e:
            logger.error(f"Error en indexación: {e}")
            return False
    
    def query(self, question: str, k: int = 5, include_sources: bool = True) -> Dict[str, Any]:
        """
        Realiza consulta RAG.
        
        Args:
            question: Pregunta del usuario
            k: Número de fragmentos a recuperar
            include_sources: Incluir fuentes en respuesta
            
        Returns:
            Diccionario con respuesta y fuentes
        """
        try:
            # 1. Generar embedding de la pregunta
            query_vector = self.embedding.generate_embedding(question)
            
            # 2. Buscar en FAISS
            results = self.vector_store.search(query_vector, k=k)
            
            if not results:
                return {"answer": "No encontré información en el repositorio.", "sources": []}
            
            # 3. Recuperar fragmentos del caché
            fragments = []
            sources = []
            
            for r in results:
                chunk_id = r["id"]
                content = self.cache.get_chunk(chunk_id)
                if content:
                    fragments.append(content)
                    # Extraer nombre del archivo del ID
                    parts = chunk_id.split(":")
                    file_name = parts[1] if len(parts) > 1 else "desconocido"
                    sources.append({"file": file_name, "score": r.get("score", 0)})
            
            if not fragments:
                return {"answer": "No se pudo recuperar el contenido de los fragmentos.", "sources": []}
            
            # 4. Construir contexto y prompt
            context = "\n\n".join(fragments[:5])
            prompt = f"Responde basado en este contexto:\n\n{context}\n\nPregunta: {question}"
            
            # 5. Generar respuesta
            answer = self.llm.generate(prompt)
            
            return {
                "answer": answer,
                "sources": sources if include_sources else []
            }
            
        except Exception as e:
            logger.error(f"Error en query: {e}")
            return {"answer": f"Error: {e}", "sources": []}
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del servicio."""
        return {
            'processing_stats': self.stats,
            'embedding_dimension': self.embedding_dimension,
            'model_used': self.model_name
        }