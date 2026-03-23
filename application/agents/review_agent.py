"""
Agente especializado en revisar código.
"""

import logging
from typing import Dict, Any, Optional

from application.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ReviewAgent(BaseAgent):
    """
    Agente que revisa código en busca de mejoras y bugs.
    """
    
    INSTRUCTIONS = """Eres un experto en calidad de código y buenas prácticas. Tu tarea es revisar el código y proporcionar retroalimentación constructiva.

                    ASPECTOS A EVALUAR:
                    1. **Legibilidad**: Nombres de variables, estructura, formato
                    2. **Mantenibilidad**: Complejidad, duplicación, modularidad
                    3. **Eficiencia**: Complejidad algorítmica, optimizaciones posibles
                    4. **Errores potenciales**: Bugs, casos borde, manejo de excepciones
                    5. **Buenas prácticas**: Patrones de diseño, principios SOLID
                    6. **Seguridad**: Inyección de código, validación de entradas
                    7. **PEP8**: Cumplimiento de estándares de estilo Python

                    FORMATO DE RESPUESTA:
                    - **Resumen**: Evaluación general
                    - **Aspectos positivos**: Lo que está bien
                    - **Áreas de mejora**: Lista con sugerencias específicas
                    - **Código sugerido**: Si aplica, muestra cómo mejorar
                    - **Prioridad**: Alta/Media/Baja para cada sugerencia"""
    
    def __init__(self):
        """Inicializa el agente de revisión."""
        super().__init__(
            name="ReviewAgent",
            description="Revisa código y sugiere mejoras"
        )
    
    def can_handle(self, query: str) -> bool:
        """
        Determina si la consulta es sobre revisión de código.
        """
        query_lower = query.lower()
        review_keywords = [
            'revisa', 'revisar', 'revisión', 'review', 'mejora',
            'optimiza', 'bug', 'error', 'problema', 'corrige',
            'código malo', 'refactoriza', 'pep8', 'estándares'
        ]
        return any(kw in query_lower for kw in review_keywords)
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Genera revisión del código.
        """
        try:
            # Recuperar contexto relevante
            fragments = self._retrieve_context(query, k=5)
            
            if not fragments:
                return {
                    'answer': "No encontré código relevante para revisar en el repositorio.",
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