"""
Agente especializado en generar documentación.
"""

import logging
from typing import Dict, Any, Optional

from application.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class DocsAgent(BaseAgent):
    """
    Agente que genera documentación automática del código.
    """
    
    INSTRUCTIONS = """Eres un experto en documentación técnica. Tu tarea es generar documentación clara y completa del código.

                    FORMATOS SOPORTADOS:
                    - **Docstrings**: Para funciones y clases (formato Google/NumPy)
                    - **README**: Documentación de alto nivel del proyecto
                    - **Comentarios**: Explicaciones en línea para código complejo
                    - **Guías**: Tutoriales y ejemplos de uso

                    CARACTERÍSTICAS DE LA DOCUMENTACIÓN:
                    - Explica el propósito del código
                    - Documenta parámetros y valores de retorno
                    - Menciona excepciones que puede lanzar
                    - Incluye ejemplos de uso cuando sea útil
                    - Sigue las convenciones del lenguaje (PEP 257 para Python)

                    FORMATO DE RESPUESTA:
                    1. **Resumen**: Propósito del código documentado
                    2. **Documentación generada**: El texto de documentación
                    3. **Recomendaciones**: Sugerencias para mejorar la documentación existente"""
    
    def __init__(self):
        """Inicializa el agente de documentación."""
        super().__init__(
            name="DocsAgent",
            description="Genera documentación automática"
        )
    
    def can_handle(self, query: str) -> bool:
        """
        Determina si la consulta es sobre documentación.
        """
        query_lower = query.lower()
        docs_keywords = [
            'documenta', 'documentación', 'docstring', 'comentario',
            'genera docs', 'escribe documentación', 'documentar'
        ]
        return any(kw in query_lower for kw in docs_keywords)
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Genera documentación del código.
        """
        try:
            # Recuperar contexto relevante
            fragments = self._retrieve_context(query, k=5)
            
            if not fragments:
                return {
                    'answer': "No encontré código para documentar en el repositorio.",
                    'sources': [],
                    'agent': self.name
                }
            
            # Construir contexto con el código completo
            context_text = self._build_context_text(fragments)
            
            # Construir prompt
            prompt = self._build_prompt(query, context_text, self.INSTRUCTIONS)
            
            # Generar respuesta
            if self.llm:
                answer = self.llm.generate(prompt)
            else:
                answer = "Error: LLM no configurado"
            
            # Preparar fuentes
            sources = [
                {
                    'file': f['file'],
                    'preview': f['content'][:300] + "..." if len(f['content']) > 300 else f['content'],
                    'score': f.get('score', 0)
                }
                for f in fragments[:3]
            ]
            
            return {
                'answer': answer,
                'sources': sources,
                'agent': self.name,
                'fragments_used': len(fragments)
            }
            
        except Exception as e:
            logger.error(f"Error en {self.name}: {e}")
            return {
                'answer': f"Error procesando la solicitud: {str(e)}",
                'sources': [],
                'agent': self.name
            }