"""
Fábrica de servicios para centralizar la creación de componentes.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from application.services.rag_gemini_service import RAGService
from application.graph.workflow import AgentWorkflow
from domain.models.repository import Repository

logger = logging.getLogger(__name__)


class ServiceFactory:
    """
    Fábrica para crear servicios de manera centralizada.
    Evita duplicación de código y centraliza la lógica de inicialización.
    """
    
    @staticmethod
    def is_api_key_valid() -> bool:
        """
        Verifica si la API key de Gemini está configurada.
        
        Returns:
            True si la API key está configurada
        """
        api_key = os.getenv("GEMINI_API_KEY")
        return api_key is not None and api_key != "" and api_key != "tu_api_key_aqui"
    
    @staticmethod
    def create_rag_service(
        repo_name: str,
        repo_path: Path,
        repo_id: int,
        prefer_pro: bool = False,
        max_file_size_mb: int = 1,
        include_docs: bool = False
    ) -> Optional[RAGService]:
        """
        Crea un servicio RAG con la configuración actual.
        Retorna None si la API key no es válida.
        
        Args:
            repo_name: Nombre del repositorio
            repo_path: Ruta física del repositorio
            repo_id: ID en base de datos
            prefer_pro: Preferir modelos Pro
            max_file_size_mb: Tamaño máximo de archivo
            include_docs: Incluir documentación
            
        Returns:
            Instancia de RAGGeminiService o None si error
        """
        if not ServiceFactory.is_api_key_valid():
            logger.warning("No se puede crear RAGService: API key no configurada")
            return None
        
        logger.info(f"Creando RAGService para {repo_name} (ID: {repo_id})")
        
        try:
            return RAGService(
                repo_name=repo_name,
                repo_path=repo_path,
                repo_id=repo_id,
                prefer_pro=prefer_pro,
                max_file_size_mb=max_file_size_mb,
                include_docs=include_docs
            )
        except Exception as e:
            logger.error(f"Error creando RAGService: {e}")
            return None
    
    @staticmethod
    def create_agent_workflow(rag_service: Optional[RAGService]) -> Optional[AgentWorkflow]:
        """
        Crea un flujo de agentes a partir de un servicio RAG.
        Retorna None si el servicio RAG no es válido.
        
        Args:
            rag_service: Servicio RAG configurado
            
        Returns:
            Instancia de AgentWorkflow o None
        """
        if rag_service is None:
            logger.warning("No se puede crear AgentWorkflow: RAGService no disponible")
            return None
        
        logger.info(f"Creando AgentWorkflow para {rag_service.repo_name}")
        
        try:
            return AgentWorkflow(rag_service)
        except Exception as e:
            logger.error(f"Error creando AgentWorkflow: {e}")
            return None
    
    @staticmethod
    def setup_repository_services(
        repo: Repository,
        prefer_pro: bool = False,
        max_file_size_mb: int = 1,
        include_docs: bool = False
    ) -> Tuple[Optional[RAGService], Optional[AgentWorkflow]]:
        """
        Configura todos los servicios para un repositorio.
        
        Args:
            repo: Repositorio (debe tener db_id)
            prefer_pro: Preferir modelos Pro
            max_file_size_mb: Tamaño máximo de archivo
            include_docs: Incluir documentación
            
        Returns:
            Tupla (rag_service, agent_workflow) - pueden ser None
        """
        rag_service = ServiceFactory.create_rag_service(
            repo_name=repo.name,
            repo_path=repo.path,
            repo_id=repo.db_id,
            prefer_pro=prefer_pro,
            max_file_size_mb=max_file_size_mb,
            include_docs=include_docs
        )
        
        agent_workflow = ServiceFactory.create_agent_workflow(rag_service) if rag_service else None
        
        return rag_service, agent_workflow