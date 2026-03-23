"""
Agente enrutador que clasifica consultas y dirige al agente adecuado.
"""

import logging
import re
from typing import Dict, Any, Optional

from application.agents.base_agent import BaseAgent
from application.agents.explain_agent import ExplainAgent
from application.agents.review_agent import ReviewAgent
from application.agents.docs_agent import DocsAgent

logger = logging.getLogger(__name__)


class RouterAgent(BaseAgent):
    """
    Agente que clasifica consultas y determina qué agente debe procesarlas.
    
    Tipos de consulta:
    - explain: Explicar código, funciones, clases
    - review: Revisar código, encontrar bugs, sugerir mejoras
    - docs: Generar documentación
    - general: Preguntas generales sobre el repositorio
    """
    
    # Palabras clave para cada tipo de consulta
    EXPLAIN_KEYWORDS = [
        'explica', 'explique', 'qué hace', 'cómo funciona', 'qué es',
        'para qué sirve', 'describe', 'definición', 'significado'
    ]
    
    REVIEW_KEYWORDS = [
        'revisa', 'revisar', 'revisión', 'review', 'mejora', 'optimiza',
        'bug', 'error', 'problema', 'fallo', 'corrige', 'arregla',
        'código malo', 'code smell', 'refactoriza', 'refactorizar'
    ]
    
    DOCS_KEYWORDS = [
        'documenta', 'documentación', 'docstring', 'comentario',
        'genera docs', 'escribe documentación', 'explicación'
    ]
    
    GENERAL_KEYWORDS = [
        'qué', 'cómo', 'dónde', 'cuándo', 'por qué', 'quién',
        'resumen', 'estructura', 'organización', 'arquitectura'
    ]
    
    def __init__(self):
        """Inicializa el agente enrutador."""
        super().__init__(
            name="RouterAgent",
            description="Clasifica consultas y dirige al agente adecuado"
        )
    
    def can_handle(self, query: str) -> bool:
        """
        El router puede manejar cualquier consulta (siempre retorna True).
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Siempre True
        """
        return True
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Clasifica la consulta y determina el tipo.
        
        Args:
            query: Consulta del usuario
            context: Contexto adicional
            
        Returns:
            Diccionario con tipo de consulta y metadatos
        """
        query_lower = query.lower()
        
        # Determinar tipo de consulta
        query_type = self._classify_query(query_lower)
        
        logger.info(f"Consulta clasificada como: {query_type}")
        
        return {
            'type': query_type,
            'original_query': query,
            'confidence': 0.8,
            'should_use_rag': query_type in ['general', 'explain']
        }
    
    def _classify_query(self, query: str) -> str:
        """
        Clasifica la consulta según palabras clave.
        
        Args:
            query: Consulta en minúsculas
            
        Returns:
            Tipo de consulta
        """
        # Verificar cada tipo
        for keyword in self.EXPLAIN_KEYWORDS:
            if keyword in query:
                return 'explain'
        
        for keyword in self.REVIEW_KEYWORDS:
            if keyword in query:
                return 'review'
        
        for keyword in self.DOCS_KEYWORDS:
            if keyword in query:
                return 'docs'
        
        # Por defecto, general
        return 'general'
    
    def get_agent_for_type(self, query_type: str):
        """
        Obtiene el agente adecuado para el tipo de consulta.
        
        Args:
            query_type: Tipo de consulta
            
        Returns:
            Instancia del agente correspondiente
        """
       
        
        agents = {
            'explain': ExplainAgent,
            'review': ReviewAgent,
            'docs': DocsAgent,
            'general': ExplainAgent  # Por defecto usa ExplainAgent
        }
        
        agent_class = agents.get(query_type, ExplainAgent)
        return agent_class()