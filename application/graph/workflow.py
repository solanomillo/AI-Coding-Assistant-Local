"""
Orquestación de agentes usando LangGraph.
"""

import logging
from typing import Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from application.agents import RouterAgent, ExplainAgent, ReviewAgent, DocsAgent
from application.services.rag_gemini_service import RAGService
from infrastructure.database.mysql_repository import MySQLRepository

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
    """
    
    STATS_KEYWORDS = [
        'cuántos archivos', 'cuantos archivos', 'archivos tiene',
        'lista los archivos', 'listar archivos', 'qué archivos',
        'resumen del repositorio', 'estadísticas', 'información del repositorio',
        'extensión', 'extensiones', 'tipos de archivo', 'lenguajes'
    ]
    
    def __init__(self, rag_service: Optional[RAGService] = None):
        """
        Inicializa el flujo de trabajo.
        
        Args:
            rag_service: Servicio RAG para contexto
        """
        self.rag_service = rag_service
        
        self.router = RouterAgent()
        self.explain_agent = ExplainAgent()
        self.review_agent = ReviewAgent()
        self.docs_agent = DocsAgent()
        
        if rag_service:
            self._configure_agents(rag_service)
        
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        
        logger.info("AgentWorkflow inicializado correctamente")
    
    def _configure_agents(self, rag_service: RAGService) -> None:
        """
        Configura todos los agentes con los servicios del RAG.
        """
        llm = rag_service.llm
        vector_store = rag_service.vector_store
        embedding_service = rag_service.embedding
        cache_service = rag_service.cache
        repo_path = rag_service.repo_path
        repo_context = {
            'name': rag_service.repo_name,
            'id': rag_service.repo_id
        }
        
        for agent in [self.explain_agent, self.review_agent, self.docs_agent]:
            agent.set_llm(llm)
            agent.set_vector_store(vector_store)
            agent.set_embedding_service(embedding_service)
            agent.set_cache_service(cache_service)
            agent.set_repo_path(repo_path)
            agent.set_repo_context(repo_context)
    
    def _build_graph(self) -> StateGraph:
        """
        Construye el grafo de agentes.
        """
        workflow = StateGraph(AgentState)
        
        workflow.add_node("router", self._router_node)
        workflow.add_node("explain", self._explain_node)
        workflow.add_node("review", self._review_node)
        workflow.add_node("docs", self._docs_node)
        workflow.add_node("general", self._general_node)
        
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
        
        workflow.add_edge("explain", END)
        workflow.add_edge("review", END)
        workflow.add_edge("docs", END)
        workflow.add_edge("general", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _router_node(self, state: AgentState) -> AgentState:
        """Nodo del router."""
        result = self.router.process(state['query'])
        return {
            **state,
            'query_type': result['type'],
            'context': result
        }
    
    def _route_to_agent(self, state: AgentState) -> str:
        """Determina qué agente usar."""
        return state.get('query_type', 'general')
    
    def _explain_node(self, state: AgentState) -> AgentState:
        """Nodo del agente de explicación."""
        result = self.explain_agent.process(state['query'])
        return {**state, 'response': result, 'sources': result.get('sources', [])}
    
    def _review_node(self, state: AgentState) -> AgentState:
        """Nodo del agente de revisión."""
        result = self.review_agent.process(state['query'])
        return {**state, 'response': result, 'sources': result.get('sources', [])}
    
    def _docs_node(self, state: AgentState) -> AgentState:
        """Nodo del agente de documentación."""
        result = self.docs_agent.process(state['query'])
        return {**state, 'response': result, 'sources': result.get('sources', [])}
    
    def _is_stats_query(self, query: str) -> bool:
        """Determina si la consulta es sobre estadísticas."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.STATS_KEYWORDS)
    
    def _get_repository_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del repositorio desde la base de datos."""
        if not self.rag_service:
            return {}
        
        try:
            db = MySQLRepository()
            repo_data = db.get_repository(self.rag_service.repo_id)
            if not repo_data:
                return {}
            
            files = db.get_files(self.rag_service.repo_id)
            
            files_by_extension = {}
            for file in files:
                ext = file.get('extension', 'unknown')
                if ext not in files_by_extension:
                    files_by_extension[ext] = []
                files_by_extension[ext].append(file.get('file_name', 'unknown'))
            
            return {
                'name': repo_data.get('name', 'desconocido'),
                'file_count': repo_data.get('file_count', 0),
                'total_lines': repo_data.get('total_lines', 0),
                'files': files,
                'files_by_extension': files_by_extension,
                'extensions': list(files_by_extension.keys())
            }
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}
    
    def _format_stats_response(self, query: str, stats: Dict[str, Any]) -> str:
        """Formatea respuesta para consultas de estadísticas."""
        query_lower = query.lower()
        
        if 'cuántos archivos' in query_lower or 'cuantos archivos' in query_lower or 'archivos tiene' in query_lower:
            file_count = stats.get('file_count', 0)
            files_by_ext = stats.get('files_by_extension', {})
            
            response = f"El repositorio **{stats.get('name', 'actual')}** contiene **{file_count}** archivos.\n\n"
            if files_by_ext:
                response += "Distribucion por tipo:\n"
                for ext, files in files_by_ext.items():
                    ext_name = ext[1:] if ext.startswith('.') else ext
                    response += f"- {ext_name}: {len(files)} archivo(s)\n"
            return response
        
        if 'lista los archivos' in query_lower or 'listar archivos' in query_lower or 'qué archivos' in query_lower:
            files = stats.get('files', [])
            files_by_ext = stats.get('files_by_extension', {})
            
            if not files:
                return "No se encontraron archivos en el repositorio."
            
            response = f"Archivos en {stats.get('name', 'el repositorio')}\n\n"
            for ext, file_list in sorted(files_by_ext.items()):
                ext_name = ext[1:] if ext.startswith('.') else ext
                response += f"{ext_name.upper()} ({len(file_list)} archivos)\n"
                for file in sorted(file_list)[:20]:
                    response += f"- {file}\n"
                if len(file_list) > 20:
                    response += f"- ... y {len(file_list) - 20} mas\n"
                response += "\n"
            return response
        
        file_count = stats.get('file_count', 0)
        total_lines = stats.get('total_lines', 0)
        files_by_ext = stats.get('files_by_extension', {})
        
        response = f"Resumen del repositorio {stats.get('name', 'actual')}\n\n"
        response += f"- Total de archivos: {file_count}\n"
        response += f"- Total de lineas: {total_lines}\n\n"
        
        if files_by_ext:
            response += "Distribucion por tipo:\n"
            for ext, files in sorted(files_by_ext.items()):
                ext_name = ext[1:] if ext.startswith('.') else ext
                response += f"- {ext_name}: {len(files)} archivo(s)\n"
        
        return response
    
    def _create_summary_agent(self):
        """
        Crea un agente temporal para generar resúmenes.
        """
        from application.agents.explain_agent import ExplainAgent
        agent = ExplainAgent()
        
        if self.rag_service:
            agent.set_repo_path(self.rag_service.repo_path)
            agent.set_cache_service(self.rag_service.cache)
            agent.set_embedding_service(self.rag_service.embedding)
            agent.set_llm(self.rag_service.llm)
        
        return agent
    
    def _generate_compact_summary(self) -> tuple:
        """
        Genera un resumen compacto del repositorio.
        
        Returns:
            Tupla (summary_text, sources)
        """
        agent = self._create_summary_agent()
        repo_summary = agent._generate_repository_summary()
        
        if not repo_summary['files']:
            return "", []
        
        compact_context = agent._build_compact_context(repo_summary)
        sources = [{'file': f['path'], 'score': 1.0} for f in repo_summary['files'][:5]]
        
        logger.info(f"Resumen generado: {repo_summary['total_files']} archivos, {len(compact_context)} caracteres")
        
        return compact_context, sources
    
    def _generate_general_response(self, query: str, context_text: str) -> str:
        """
        Genera respuesta para consulta general.
        """
        instructions = """Eres un experto en analisis de codigo. Tu tarea es proporcionar un resumen claro y completo del repositorio.

                    CARACTERISTICAS:
                    - Explica que hace el repositorio en general
                    - Menciona los tipos de archivos presentes (HTML, CSS, JavaScript, etc.)
                    - Describe la estructura y funcionalidad principal
                    - Si hay archivos de diferentes lenguajes, mencionarlos todos
                    - NO muestres codigo, solo describe la funcionalidad
                    - Ser conciso pero completo
                    - Responder en español"""

        agent = self._create_summary_agent()
        prompt = agent._build_prompt(query, context_text, instructions)
        
        if self.rag_service and self.rag_service.llm:
            return self.rag_service.llm.generate(prompt)
        
        return "No se pudo generar respuesta. Verifica la configuracion de la API."
    
    def _general_node(self, state: AgentState) -> AgentState:
        """
        Nodo para consultas generales.
        """
        query = state['query']
        
        # Verificar si es consulta de estadísticas
        if self._is_stats_query(query):
            stats = self._get_repository_stats()
            if stats:
                answer = self._format_stats_response(query, stats)
                return {
                    **state,
                    'response': {
                        'answer': answer,
                        'sources': [],
                        'agent': 'StatisticsAgent'
                    },
                    'sources': []
                }
        
        # Para consultas generales, generar resumen compacto
        if self.rag_service:
            try:
                compact_context, sources = self._generate_compact_summary()
                
                if compact_context:
                    answer = self._generate_general_response(query, compact_context)
                    return {
                        **state,
                        'response': {
                            'answer': answer,
                            'sources': sources,
                            'agent': 'GeneralAgent'
                        },
                        'sources': sources
                    }
            except Exception as e:
                logger.error(f"Error generando resumen: {e}")
        
        # Fallback: usar RAG normal
        if self.rag_service:
            result = self.rag_service.query(query, include_sources=True)
            return {
                **state,
                'response': {
                    'answer': result['answer'],
                    'sources': result.get('sources', []),
                    'agent': 'RAGService'
                },
                'sources': result.get('sources', [])
            }
        
        return {
            **state,
            'response': {
                'answer': "No hay servicio RAG disponible.",
                'sources': [],
                'agent': 'general'
            }
        }
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        Procesa una consulta a través del grafo de agentes.
        """
        try:
            logger.info(f"Procesando consulta: {query[:100]}...")
            
            initial_state: AgentState = {
                'query': query,
                'query_type': '',
                'context': None,
                'response': None,
                'error': None,
                'sources': []
            }
            
            config = {"configurable": {"thread_id": "1"}}
            final_state = self.graph.invoke(initial_state, config)
            
            response = final_state.get('response', {})
            logger.info(f"Consulta procesada por: {response.get('agent', 'desconocido')}")
            
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
        """
        return {
            'router': self.router.description,
            'explain': self.explain_agent.description,
            'review': self.review_agent.description,
            'docs': self.docs_agent.description
        }