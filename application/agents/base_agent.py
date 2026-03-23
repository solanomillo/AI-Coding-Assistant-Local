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
        logger.info(f"Agente '{name}' inicializado: {description}")
    
    def set_llm(self, llm) -> None:
        """
        Establece el cliente LLM para el agente.
        
        Args:
            llm: Cliente LLM (GeminiLLM)
        """
        self.llm = llm
        logger.debug(f"LLM configurado para agente {self.name}")
    
    def set_vector_store(self, vector_store) -> None:
        """
        Establece el vector store para búsqueda de contexto.
        
        Args:
            vector_store: Almacén vectorial FAISS
        """
        self.vector_store = vector_store
        logger.debug(f"Vector store configurado para agente {self.name}")
    
    def set_embedding_service(self, embedding_service) -> None:
        """
        Establece el servicio de embeddings.
        
        Args:
            embedding_service: Servicio de embeddings
        """
        self.embedding_service = embedding_service
        logger.debug(f"Embedding service configurado para agente {self.name}")
    
    def set_repo_context(self, repo_context: Dict[str, Any]) -> None:
        """
        Establece el contexto del repositorio.
        
        Args:
            repo_context: Diccionario con información del repositorio
        """
        self.repo_context = repo_context
        logger.debug(f"Contexto de repositorio configurado para agente {self.name}")
    
    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """
        Determina si este agente puede manejar la consulta.
        
        Args:
            query: Consulta del usuario
            
        Returns:
            True si el agente puede manejar la consulta
        """
        pass
    
    @abstractmethod
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Procesa la consulta y genera respuesta.
        
        Args:
            query: Consulta del usuario
            context: Contexto adicional opcional
            
        Returns:
            Diccionario con respuesta y metadatos
        """
        pass
    
    def _retrieve_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Recupera contexto relevante del vector store.
        
        Args:
            query: Consulta para búsqueda
            k: Número de resultados
            
        Returns:
            Lista de fragmentos relevantes
        """
        if not self.vector_store:
            logger.warning(f"Vector store no disponible para agente {self.name}")
            return []
        
        if not self.embedding_service:
            logger.warning(f"Embedding service no disponible para agente {self.name}")
            return []
        
        try:
            # CORRECCIÓN: Generar embedding de la consulta
            logger.debug(f"Generando embedding para consulta: {query[:50]}...")
            query_vector = self.embedding_service.generate_embedding(query)
            
            # Verificar dimensión
            if len(query_vector) != self.embedding_service.get_dimension():
                logger.error(f"Dimensión incorrecta: {len(query_vector)}")
                return []
            
            logger.debug(f"Embedding generado: {len(query_vector)} dimensiones")
            
            # Buscar en FAISS
            results = self.vector_store.search(query_vector, k=k)
            logger.debug(f"Recuperados {len(results)} fragmentos para {self.name}")
            return results
            
        except Exception as e:
            logger.error(f"Error recuperando contexto: {e}")
            return []
    
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
                    - Sé técnico y preciso
                    - Usa formato de código con ``` cuando sea necesario
                    - Responde en español
                    - Si no hay suficiente información, indícalo claramente

                    RESPUESTA:"""