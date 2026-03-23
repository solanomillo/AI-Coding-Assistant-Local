"""
Agente especializado en explicar código.
"""

import logging
from typing import Dict, Any, Optional

from application.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ExplainAgent(BaseAgent):
    """
    Agente que explica funciones, clases y bloques de código.
    """
    
    INSTRUCTIONS = """Eres un experto en análisis de código. Tu tarea es explicar el código de manera clara y didáctica.

                    CARACTERÍSTICAS:
                    - Explica qué hace el código, no solo qué dice
                    - Menciona los parámetros y valores de retorno
                    - Explica la lógica y el flujo de ejecución
                    - Si hay partes complejas, destácalas y explica por qué son así
                    - Usa ejemplos cuando sea útil
                    - Mantén un tono educativo pero profesional

                    FORMATO DE RESPUESTA:
                    1. Resumen breve de la función/clase
                    2. Explicación detallada
                    3. Parámetros (si aplica)
                    4. Valor de retorno (si aplica)
                    5. Ejemplo de uso (si es útil)"""
    
    def __init__(self):
        """Inicializa el agente de explicación."""
        super().__init__(
            name="ExplainAgent",
            description="Explica código, funciones y clases"
        )
    
    def can_handle(self, query: str) -> bool:
        """
        Determina si la consulta es sobre explicación de código.
        """
        query_lower = query.lower()
        explain_keywords = [
            'explica', 'explique', 'qué hace', 'cómo funciona',
            'para qué sirve', 'describe', 'qué es', 'paso a paso'
        ]
        return any(kw in query_lower for kw in explain_keywords)
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Genera explicación del código.
        """
        try:
            # Recuperar contexto relevante
            fragments = self._retrieve_context(query, k=5)
            
            if not fragments:
                return {
                    'answer': "No encontré código relevante en el repositorio para explicar.",
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