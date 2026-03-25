import logging
import time
from typing import List, Dict, Any
from pathlib import Path
from infrastructure.embeddings.gemini_embedding import GeminiEmbedding
from infrastructure.vector_db.faiss_store import FAISSStore
from infrastructure.llm_clients.gemini_llm import GeminiLLM
from domain.models.repository import Repository
from application.services.cache_service import CacheService

logger = logging.getLogger(__name__)


class RAGService:

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
        prefer_pro: bool = False,
        max_file_size_mb: int = 1,
        include_docs: bool = False
    ):
        self.repo_name = repo_name
        self.repo_path = repo_path
        self.repo_id = repo_id
        self.prefer_pro = prefer_pro
        self.max_file_size_mb = max_file_size_mb
        self.include_docs = include_docs

        self.embedding = GeminiEmbedding()
        self.llm = GeminiLLM(prefer_pro=prefer_pro)
        self.cache = CacheService()

        self.embedding_dimension = self.embedding.get_dimension()

        safe_name = repo_name.replace(" ", "_")
        index_path = Path(f"data/vectors/{safe_name}.index")
        index_path.parent.mkdir(parents=True, exist_ok=True)

        self.vector_store = FAISSStore(
            dimension=self.embedding_dimension,
            index_path=index_path
        )

        self.stats = {
            "total_chunks": 0,
            "chunks_processed": 0,
            "api_calls": 0,
            "cache_hits": 0,
        }

        logger.info("RAGService optimizado inicializado")

    # -------------------------
    # CHUNKING
    # -------------------------
    def _chunk_code(self, text: str) -> List[str]:
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.MAX_CHUNK_SIZE
            chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk)

            start = end - self.CHUNK_OVERLAP

            if len(chunks) >= self.MAX_FRAGMENTS_PER_FILE:
                break

        return chunks

    # -------------------------
    # INDEXACIÓN
    # -------------------------
    def index_repository(self, repository: Repository) -> bool:
        start_time = time.time()

        try:
            all_ids = []

            # 1. Crear chunks
            for file in repository.files:
                try:
                    file_path = self.repo_path / file.relative_path
                    if not file_path.exists():
                        continue

                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    chunks = self._chunk_code(content)

                    for i, chunk in enumerate(chunks):
                        chunk_id = f"{self.repo_id}:{file.relative_path}:{i}"

                        self.cache.put_chunk(chunk_id, chunk)
                        all_ids.append(chunk_id)

                except Exception as e:
                    logger.error(f"Error archivo {file.relative_path}: {e}")

            self.stats["total_chunks"] = len(all_ids)

            if not all_ids:
                return False

            # 2. EMBEDDINGS (BATCH REAL)
            final_vectors = []
            final_ids = []

            for i in range(0, len(all_ids), self.BATCH_SIZE):
                batch_ids = all_ids[i:i + self.BATCH_SIZE]

                batch_texts = []
                valid_ids = []

                # preparar batch
                for chunk_id in batch_ids:
                    cached_vector = self.cache.get_embedding(chunk_id)

                    if cached_vector:
                        self.stats["cache_hits"] += 1
                        final_vectors.append(cached_vector)
                        final_ids.append(chunk_id)
                        continue

                    content = self.cache.get_chunk(chunk_id)
                    if content:
                        batch_texts.append(content)
                        valid_ids.append(chunk_id)

                if not batch_texts:
                    continue

                try:
                    # UNA SOLA LLAMADA
                    vectors = self.embedding.generate_embeddings_batch(batch_texts)
                    self.stats["api_calls"] += 1

                    for chunk_id, vector in zip(valid_ids, vectors):
                        if vector and len(vector) == self.embedding_dimension:
                            final_vectors.append(vector)
                            final_ids.append(chunk_id)

                            # guardar embedding en cache
                            self.cache.put_embedding(chunk_id, vector)

                            self.stats["chunks_processed"] += 1

                except Exception as e:
                    logger.error(f"Error batch embeddings: {e}")

                if i + self.BATCH_SIZE < len(all_ids):
                    time.sleep(self.BATCH_DELAY)

            # 3. Guardar en FAISS
            if final_vectors:
                self.vector_store.add_vectors(final_vectors, final_ids)

                elapsed = time.time() - start_time

                logger.info("INDEXACIÓN COMPLETADA")
                logger.info(f"Chunks: {self.stats['chunks_processed']}")
                logger.info(f"API Calls: {self.stats['api_calls']}")
                logger.info(f"Tiempo: {elapsed:.2f}s")

                return True

            return False

        except Exception as e:
            logger.error(f"Error indexando: {e}")
            return False

    # -------------------------
    # QUERY
    # -------------------------
    def query(self, question: str, k: int = 5, include_sources: bool = True) -> Dict[str, Any]:
        try:
            query_vector = self.embedding.generate_embedding(question)

            results = self.vector_store.search(query_vector, k=k)

            if not results:
                return {"answer": "No encontré información en el repositorio.", "sources": []}

            fragments = []
            sources = []

            for r in results:
                chunk_id = r["id"]
                content = self.cache.get_chunk(chunk_id)
                if content:
                    fragments.append(content)
                    # Extraer nombre del archivo del ID
                    # Formato: repo_id:ruta:indice
                    parts = chunk_id.split(":")
                    file_name = parts[1] if len(parts) > 1 else "desconocido"
                    
                    sources.append({
                        "file": file_name,
                        "score": r.get("score", 0)
                    })

            context = "\n\n".join(fragments[:5])

            prompt = f"""
                    Responde basado en este contexto:

                    {context}

                    Pregunta: {question}
                    """

            answer = self.llm.generate(prompt)

            return {
                "answer": answer,
                "sources": sources if include_sources else []
            }

        except Exception as e:
            logger.error(f"Error en query: {e}")
            return {
                "answer": f"Error: {e}",
                "sources": []
            }

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del servicio."""
        return {
            'processing_stats': self.stats,
            'embedding_dimension': self.embedding_dimension
        }