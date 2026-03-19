"""
Servicio RAG con Gemini + FAISS.
ARQUITECTURA CORREGIDA - FASE 3:
- Usa rutas permanentes de data/repositories/
- Integración perfecta con repo_service.py
- Manejo robusto de errores de archivos
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
from infrastructure.embeddings.gemini_embedding import GeminiEmbedding
from infrastructure.vector_db.faiss_store import FAISSStore
from infrastructure.llm_clients.gemini_llm import GeminiLLM
from domain.models.repository import Repository

logger = logging.getLogger(__name__)


class RAGGeminiService:
    """
    Servicio RAG profesional con Gemini + FAISS.
    
    RESPONSABILIDADES:
    1. Indexar repositorios usando embeddings de Gemini
    2. Almacenar vectores en FAISS
    3. Responder preguntas sobre el código
    4. Mantener contexto de conversación
    
    NOTA: Asume que los archivos YA ESTÁN en disco permanentemente
          (gestionados por RepositoryService)
    """
    
    def __init__(
        self,
        repo_name: str,
        repo_path: Path,
        prefer_pro: bool = False,
        auto_fallback: bool = True
    ):
        """
        Inicializa servicio RAG.
        
        Args:
            repo_name: Nombre del repositorio
            repo_path: Ruta física donde están los archivos
            prefer_pro: Preferir modelos Pro de Gemini
            auto_fallback: Fallback automático si falla un modelo
        """
        # Validar que la ruta existe
        if not repo_path.exists():
            raise ValueError(f"La ruta del repositorio no existe: {repo_path}")
        
        self.repo_name = repo_name
        self.repo_path = repo_path
        
        # Inicializar componentes
        self.embedding = GeminiEmbedding()
        self.llm = GeminiLLM(
            prefer_pro=prefer_pro,
            auto_fallback=auto_fallback
        )
        
        # Configurar índice FAISS específico para este repo
        safe_name = repo_name.replace(' ', '_').replace('/', '_')
        index_path = Path(f"data/vector_store/{safe_name}.index")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.vector_store = FAISSStore(
            dimension=self.embedding.get_dimension(),
            index_path=index_path
        )
        
        # Obtener información del modelo
        model_info = self.llm.get_model_info()
        
        logger.info("=" * 50)
        logger.info(f"✅ RAGGeminiService inicializado")
        logger.info(f"📁 Repositorio: {repo_name}")
        logger.info(f"📁 Ruta física: {repo_path}")
        logger.info(f"🤖 Modelo activo: {model_info['current_model']} ({model_info['model_type']})")
        logger.info(f"🔍 Índice FAISS: {index_path}")
        logger.info("=" * 50)
    
    def index_repository(self, repository: Repository) -> bool:
        """
        Indexa repositorio completo en FAISS.
        VERSIÓN CORREGIDA - Sin duplicación de rutas.
        """
        try:
            logger.info(f"📚 Iniciando indexación de: {repository.name}")
            logger.info(f"📁 Ruta base del repo: {self.repo_path}")
            
            all_vectors = []
            all_ids = []
            all_metadatas = []
            
            total_files = len(repository.files)
            files_processed = 0
            chunks_generated = 0
            
            for idx, file in enumerate(repository.files, 1):
                logger.info(f"📄 Procesando [{idx}/{total_files}]: {file.name}")
                
                # 🔥 CORRECCIÓN: Construir ruta correcta SIN duplicar
                # Usar self.repo_path como base + file.relative_path
                file_path = self.repo_path / file.relative_path
                logger.debug(f"Buscando archivo en: {file_path}")
                
                if not file_path.exists():
                    logger.warning(f"⚠️ No encontrado en ruta principal: {file_path}")
                    
                    # Intentar con solo el nombre del archivo
                    alt_path = self.repo_path / file.name
                    if alt_path.exists():
                        logger.info(f"✅ Encontrado en: {alt_path}")
                        file_path = alt_path
                    else:
                        # Intentar buscando en subdirectorios
                        found = False
                        for root, dirs, files in os.walk(self.repo_path):
                            if file.name in files:
                                file_path = Path(root) / file.name
                                logger.info(f"✅ Encontrado por búsqueda: {file_path}")
                                found = True
                                break
                        
                        if not found:
                            logger.error(f"❌ Archivo no encontrado: {file.name}")
                            continue
                
                try:
                    # Leer archivo
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if not content.strip():
                        logger.warning(f"⚠️ Archivo vacío: {file_path}")
                        continue
                    
                    # Dividir en fragmentos
                    chunks = GeminiEmbedding.chunk_code(content)
                    chunks_generated += len(chunks)
                    
                    # Procesar cada fragmento
                    for i, chunk in enumerate(chunks):
                        # Generar embedding
                        vector = self.embedding.generate_embedding(chunk)
                        
                        # Crear ID único
                        chunk_id = GeminiEmbedding.create_chunk_id(
                            str(file.relative_path),
                            i,
                            str(hash(chunk[:100]))
                        )
                        
                        # Metadatos
                        preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
                        
                        # 🔥 IMPORTANTE: Guardar ruta relativa, no absoluta
                        relative_path = str(file.relative_path)
                        
                        metadata = {
                            'repo': repository.name,
                            'file': relative_path,
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
                    
                except Exception as e:
                    logger.error(f"❌ Error procesando {file_path}: {e}")
                    continue
            
            # Guardar en FAISS
            if all_vectors:
                self.vector_store.add_vectors(all_vectors, all_ids, all_metadatas)
                logger.info(f"✅ Indexación completada:")
                logger.info(f"   - Archivos procesados: {files_processed}/{total_files}")
                logger.info(f"   - Fragmentos generados: {len(all_vectors)}")
                return True
            else:
                logger.error("❌ No se generaron vectores")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error en indexación: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def query(
        self,
        question: str,
        k: int = 5,
        include_sources: bool = True,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Realiza consulta RAG.
        
        Args:
            question: Pregunta del usuario
            k: Número de fragmentos a recuperar
            include_sources: Incluir fuentes en respuesta
            temperature: Temperatura para generación
            
        Returns:
            Respuesta con fuentes y metadatos
        """
        try:
            logger.info(f"🔍 Consulta: {question[:100]}...")
            
            # 1. Generar embedding de la pregunta
            query_vector = self.embedding.generate_embedding(question)
            
            # 2. Buscar en FAISS
            results = self.vector_store.search(query_vector, k=k)
            
            if not results:
                return {
                    'answer': "No encontré información relevante en el repositorio.",
                    'sources': [],
                    'model_used': self.llm.current_model
                }
            
            # 3. Construir contexto
            context_parts = []
            sources_for_display = []
            
            for i, r in enumerate(results, 1):
                metadata = r['metadata']
                
                context_parts.append(
                    f"[Fuente {i} - Archivo: {metadata['file']}]\n"
                    f"{metadata.get('preview', '')}\n"
                )
                
                sources_for_display.append({
                    'file': metadata['file'],
                    'preview': metadata.get('preview', ''),
                    'score': round(r.get('score', 0), 3)
                })
            
            context = "\n---\n".join(context_parts)
            
            # 4. Generar respuesta
            prompt = self._build_prompt(question, context)
            answer = self.llm.generate(prompt, temperature=temperature)
            
            # 5. Obtener info del modelo
            model_info = self.llm.get_model_info()
            
            return {
                'answer': answer,
                'sources': sources_for_display if include_sources else [],
                'model_used': model_info['current_model'],
                'model_type': model_info['model_type']
            }
            
        except Exception as e:
            logger.error(f"❌ Error en consulta: {e}")
            return {
                'answer': f"Error procesando consulta: {str(e)}",
                'sources': []
            }
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Construye prompt optimizado para Gemini."""
        prompt = f"""Eres un experto en análisis de código. Responde basándote ÚNICAMENTE en el contexto proporcionado.

CONTEXTO DEL REPOSITORIO:
{context}

PREGUNTA: {question}

INSTRUCCIONES:
- Responde SOLO con información del contexto
- Si la información no está en el contexto, indícalo claramente
- Sé técnico y preciso
- Usa formato de código con ```python cuando sea necesario
- Si encuentras errores potenciales, menciónalos constructivamente

RESPUESTA:"""
        
        return prompt
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Estadísticas del servicio RAG.
        
        Returns:
            Diccionario con estadísticas
        """
        vector_stats = self.vector_store.get_stats()
        model_info = self.llm.get_model_info()
        
        return {
            'repo_name': self.repo_name,
            'repo_path': str(self.repo_path),
            'vector_store': vector_stats,
            'llm': model_info,
            'embedding_dimension': self.embedding.get_dimension()
        }