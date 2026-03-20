"""
Orquestación de agentes usando LangGraph.
Define el flujo de trabajo para procesar consultas.
"""

import logging
from typing import Dict, Any, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver

from application.agents import RouterAgent, ExplainAgent, ReviewAgent, DocsAgent
from application.services.rag_gemini_service import RAGGeminiService

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    Estado del grafo de agentes.
    """
    query: str
    query_type: str
    context: Optional[Dict[str, Any]]
    response: Optional[Dict[str, Any]]
    error: Optional[str]
    sources: list


class AgentWorkflow:
    """
    Orquestador de agentes usando LangGraph.
    
    Define el flujo:
    1. Router clasifica la consulta
    2. Se dirige al agente especializado
    3. El agente procesa y genera respuesta
    """
    
    def __init__(self, rag_service: Optional[RAGGeminiService] = None):
        """
        Inicializa el flujo de trabajo.
        
        Args:
            rag_service: Servicio RAG para contexto
        """
        self.rag_service = rag_service
        
        # Inicializar agentes
        self.router = RouterAgent()
        self.explain_agent = ExplainAgent()
        self.review_agent = ReviewAgent()
        self.docs_agent = DocsAgent()
        
        # Configurar LLM y vector store para los agentes
        if rag_service:
            llm = rag_service.llm
            vector_store = rag_service.vector_store
            repo_context = {
                'name': rag_service.repo_name,
                'id': rag_service.repo_id
            }
            
            for agent in [self.explain_agent, self.review_agent, self.docs_agent]:
                agent.set_llm(llm)
                agent.set_vector_store(vector_store)
                agent.set_repo_context(repo_context)
        
        # Construir grafo
        self.graph = self._build_graph()
        self.memory = MemorySaver()
        
        logger.info("AgentWorkflow inicializado correctamente")
    
    def _build_graph(self) -> StateGraph:
        """
        Construye el grafo de agentes.
        
        Returns:
            StateGraph configurado
        """
        # Crear grafo
        workflow = StateGraph(AgentState)
        
        # Definir nodos
        workflow.add_node("router", self._router_node)
        workflow.add_node("explain", self._explain_node)
        workflow.add_node("review", self._review_node)
        workflow.add_node("docs", self._docs_node)
        workflow.add_node("general", self._general_node)
        
        # Definir flujo
        workflow.set_entry_point("router")
        
        workflow.add_conditional_edges(
            "router",
            self._route_to_agent,
            {
                "explain": "explain",
                "review": "review",
                "docs": "docs",
                "general": "general"
            }
        )
        
        # Todos los agentes terminan
        workflow.add_edge("explain", END)
        workflow.add_edge("review", END)
        workflow.add_edge("docs", END)
        workflow.add_edge("general", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _router_node(self, state: AgentState) -> AgentState:
        """
        Nodo del router: clasifica la consulta.
        
        Args:
            state: Estado actual
            
        Returns:
            Estado actualizado
        """
        result = self.router.process(state['query'])
        
        return {
            **state,
            'query_type': result['type'],
            'context': result
        }
    
    def _route_to_agent(self, state: AgentState) -> str:
        """
        Determina qué agente usar según el tipo de consulta.
        
        Args:
            state: Estado actual
            
        Returns:
            Nombre del nodo destino
        """
        query_type = state.get('query_type', 'general')
        return query_type
    
    def _explain_node(self, state: AgentState) -> AgentState:
        """
        Nodo del agente de explicación.
        
        Args:
            state: Estado actual
            
        Returns:
            Estado con respuesta
        """
        result = self.explain_agent.process(state['query'])
        return {**state, 'response': result, 'sources': result.get('sources', [])}
    
    def _review_node(self, state: AgentState) -> AgentState:
        """
        Nodo del agente de revisión.
        
        Args:
            state: Estado actual
            
        Returns:
            Estado con respuesta
        """
        result = self.review_agent.process(state['query'])
        return {**state, 'response': result, 'sources': result.get('sources', [])}
    
    def _docs_node(self, state: AgentState) -> AgentState:
        """
        Nodo del agente de documentación.
        
        Args:
            state: Estado actual
            
        Returns:
            Estado con respuesta
        """
        result = self.docs_agent.process(state['query'])
        return {**state, 'response': result, 'sources': result.get('sources', [])}
    
    def _general_node(self, state: AgentState) -> AgentState:
        """
        Nodo para consultas generales (usa RAG directamente).
        
        Args:
            state: Estado actual
            
        Returns:
            Estado con respuesta
        """
        if self.rag_service:
            result = self.rag_service.query(state['query'], include_sources=True)
            return {
                **state,
                'response': {
                    'answer': result['answer'],
                    'sources': result['sources'],
                    'agent': 'RAGService'
                },
                'sources': result.get('sources', [])
            }
        else:
            return {
                **state,
                'response': {
                    'answer': "No hay servicio RAG disponible para consultas generales.",
                    'sources': [],
                    'agent': 'general'
                }
            }
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        Procesa una consulta a través del grafo de agentes.
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Diccionario con respuesta y metadatos
        """
        try:
            logger.info(f"Procesando consulta con agentes: {query[:100]}...")
            
            # Estado inicial
            initial_state: AgentState = {
                'query': query,
                'query_type': '',
                'context': None,
                'response': None,
                'error': None,
                'sources': []
            }
            
            # Ejecutar grafo
            config = {"configurable": {"thread_id": "1"}}
            final_state = self.graph.invoke(initial_state, config)
            
            response = final_state.get('response', {})
            logger.info(f"Consulta procesada por agente: {response.get('agent', 'desconocido')}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error procesando consulta: {e}")
            return {
                'answer': f"Error procesando la consulta: {str(e)}",
                'sources': [],
                'agent': 'error'
            }
    
    def get_available_agents(self) -> Dict[str, str]:
        """
        Obtiene información de los agentes disponibles.
        
        Returns:
            Diccionario con nombres y descripciones
        """
        return {
            'router': self.router.description,
            'explain': self.explain_agent.description,
            'review': self.review_agent.description,
            'docs': self.docs_agent.description
        }