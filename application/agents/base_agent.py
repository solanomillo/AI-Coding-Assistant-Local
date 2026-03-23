"""
Clase base para todos los agentes.
Define la interfaz común y funcionalidades compartidas.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Clase base abstracta para todos los agentes.
    """
    
    def __init__(self, name: str, description: str):
        """
        Inicializa el agente base.
        
        Args:
            name: Nombre del agente
            description: Descripción de su función
        """
        self.name = name
        self.description = description
        self.llm = None
        self.vector_store = None
        self.repo_context = None
        self.embedding_service = None
        self.cache_service = None
        logger.info(f"Agente '{name}' inicializado: {description}")
    
    def set_llm(self, llm) -> None:
        """Establece el cliente LLM para el agente."""
        self.llm = llm
        logger.debug(f"LLM configurado para agente {self.name}")
    
    def set_vector_store(self, vector_store) -> None:
        """Establece el vector store para búsqueda de contexto."""
        self.vector_store = vector_store
        logger.debug(f"Vector store configurado para agente {self.name}")
    
    def set_embedding_service(self, embedding_service) -> None:
        """Establece el servicio de embeddings."""
        self.embedding_service = embedding_service
        logger.debug(f"Embedding service configurado para agente {self.name}")
    
    def set_cache_service(self, cache_service) -> None:
        """Establece el servicio de caché para recuperar fragmentos completos."""
        self.cache_service = cache_service
        logger.debug(f"Cache service configurado para agente {self.name}")
    
    def set_repo_context(self, repo_context: Dict[str, Any]) -> None:
        """Establece el contexto del repositorio."""
        self.repo_context = repo_context
        logger.debug(f"Contexto de repositorio configurado para agente {self.name}")
    
    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """Determina si este agente puede manejar la consulta."""
        pass
    
    @abstractmethod
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Procesa la consulta y genera respuesta."""
        pass
    
    def _retrieve_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Recupera contexto relevante del vector store y caché.
        
        Args:
            query: Consulta para búsqueda
            k: Número de resultados
            
        Returns:
            Lista de fragmentos con contenido completo
        """
        if not self.vector_store:
            logger.warning(f"Vector store no disponible para agente {self.name}")
            return []
        
        if not self.embedding_service:
            logger.warning(f"Embedding service no disponible para agente {self.name}")
            return []
        
        if not self.cache_service:
            logger.warning(f"Cache service no disponible para agente {self.name}")
            return []
        
        try:
            # Generar embedding de la consulta
            logger.debug(f"Generando embedding para consulta: {query[:50]}...")
            query_vector = self.embedding_service.generate_embedding(query)
            
            if len(query_vector) != self.embedding_service.get_dimension():
                logger.error(f"Dimensión incorrecta: {len(query_vector)}")
                return []
            
            # Buscar en FAISS (retorna solo IDs y scores)
            results = self.vector_store.search(query_vector, k=k)
            
            if not results:
                logger.debug("No se encontraron resultados")
                return []
            
            # Recuperar fragmentos completos del caché
            fragments = []
            for r in results:
                chunk_id = r['id']
                content = self.cache_service.get_chunk(chunk_id)
                if content:
                    # Extraer nombre del archivo del ID
                    parts = chunk_id.split(':')
                    file_name = parts[1] if len(parts) > 1 else 'desconocido'
                    
                    fragments.append({
                        'id': chunk_id,
                        'file': file_name,
                        'content': content,
                        'score': r['score']
                    })
            
            logger.debug(f"Recuperados {len(fragments)} fragmentos para {self.name}")
            return fragments
            
        except Exception as e:
            logger.error(f"Error recuperando contexto: {e}")
            return []
    
    def _build_context_text(self, fragments: List[Dict[str, Any]]) -> str:
        """
        Construye texto de contexto a partir de fragmentos.
        
        Args:
            fragments: Lista de fragmentos recuperados
            
        Returns:
            Texto de contexto formateado
        """
        context_parts = []
        for i, f in enumerate(fragments[:5]):
            context_parts.append(
                f"[Fragmento {i+1} - Archivo: {f['file']}]\n"
                f"{f['content']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str, instructions: str) -> str:
        """
        Construye prompt para el LLM.
        
        Args:
            query: Consulta del usuario
            context: Contexto recuperado
            instructions: Instrucciones específicas del agente
            
        Returns:
            Prompt completo
        """
        repo_info = ""
        if self.repo_context:
            repo_info = f"Repositorio: {self.repo_context.get('name', 'desconocido')}\n"
        
        return f"""{instructions}

                    {repo_info}
                    CONTEXTO DEL CÓDIGO:
                    {context}

                    CONSULTA DEL USUARIO:
                    {query}

                    INSTRUCCIONES ADICIONALES:
                    - Analiza el código proporcionado en el contexto
                    - Responde basándote ÚNICAMENTE en el código que ves
                    - Si el código no está presente, indícalo claramente
                    - Sé técnico y preciso
                    - Usa formato de código con ``` cuando sea necesario
                    - Responde en español

                    RESPUESTA:"""