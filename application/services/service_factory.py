"""
Fábrica de servicios para centralizar la creación de componentes.
Optimizada con caché agresivo y manejo centralizado de errores.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from application.services.rag_gemini_service import RAGService
from application.graph.workflow import AgentWorkflow
from domain.models.repository import Repository
from infrastructure.llm_clients.error_handler import APIErrorHandler

logger = logging.getLogger(__name__)


class ServiceFactory:
    """
    Fábrica para crear servicios de manera centralizada.
    """
    
    # Constantes de caché
    CACHE_DURATION_OK = 300          # 5 minutos si está OK
    CACHE_DURATION_EXHAUSTED = 3600  # 1 hora si está agotada
    MODELS_CACHE_DURATION = 3600     # 1 hora para modelos
    
    # Cache de estado de API
    _api_status_cache = {
        'available': None,
        'status': None,
        'last_check': 0
    }
    
    # Cache de modelos disponibles
    _available_models_cache = None
    _models_cache_timestamp = 0
    
    @staticmethod
    def is_api_key_valid() -> bool:
        """Verifica si la API key está configurada."""
        api_key = os.getenv("GEMINI_API_KEY")
        return api_key is not None and api_key != "" and api_key != "tu_api_key_aqui"
    
    @staticmethod
    def clear_cache() -> None:
        """Limpia todos los cachés."""
        ServiceFactory._api_status_cache = {
            'available': None,
            'status': None,
            'last_check': 0
        }
        ServiceFactory._available_models_cache = None
        ServiceFactory._models_cache_timestamp = 0
        logger.info("Cachés limpiados")
    
    @staticmethod
    def _is_api_cache_valid() -> bool:
        """Verifica si el caché de estado de API es válido."""
        cache = ServiceFactory._api_status_cache
        if cache['available'] is None:
            return False
        
        cache_age = time.time() - cache['last_check']
        
        if cache['available']:
            return cache_age < ServiceFactory.CACHE_DURATION_OK
        else:
            return cache_age < ServiceFactory.CACHE_DURATION_EXHAUSTED
    
    @staticmethod
    def _update_api_cache(available: bool, status: str) -> None:
        """Actualiza el caché de estado de API."""
        ServiceFactory._api_status_cache = {
            'available': available,
            'status': status,
            'last_check': time.time()
        }
    
    @staticmethod
    def get_available_models(force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Obtiene la lista de modelos disponibles.
        
        Args:
            force_refresh: Si True, ignora caché y consulta API
            
        Returns:
            Lista de modelos disponibles
        """
        now = time.time()
        
        # Usar caché si es válido
        if not force_refresh and ServiceFactory._available_models_cache is not None:
            cache_age = now - ServiceFactory._models_cache_timestamp
            if cache_age < ServiceFactory.MODELS_CACHE_DURATION:
                logger.debug(f"Usando caché de modelos (edad: {cache_age:.1f}s)")
                return ServiceFactory._available_models_cache
        
        # Si no hay API key válida, devolver modelos por defecto
        if not ServiceFactory.is_api_key_valid():
            return ServiceFactory._get_default_models()
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            all_models = genai.list_models()
            
            models = []
            for model in all_models:
                if 'generateContent' in model.supported_generation_methods:
                    model_name = model.name.replace('models/', '')
                    models.append({
                        'name': model_name,
                        'display_name': model.display_name or model_name,
                        'type': ServiceFactory._classify_model(model_name)
                    })
            
            models.sort(key=lambda x: (0 if 'flash' in x['name'].lower() else 1, x['name']))
            
            ServiceFactory._available_models_cache = models
            ServiceFactory._models_cache_timestamp = now
            
            logger.info(f"Modelos obtenidos: {len(models)}")
            return models
            
        except Exception as e:
            logger.error(f"Error obteniendo modelos: {e}")
            return ServiceFactory._get_default_models()
    
    @staticmethod
    def _get_default_models() -> List[Dict[str, Any]]:
        """Modelos por defecto si falla la API."""
        return [
            {'name': 'gemini-2.5-flash', 'display_name': 'Gemini 2.5 Flash', 'type': 'flash'},
            {'name': 'gemini-2.0-flash-001', 'display_name': 'Gemini 2.0 Flash', 'type': 'flash'},
            {'name': 'gemini-2.5-flash-lite', 'display_name': 'Gemini 2.5 Flash-Lite', 'type': 'flash'},
            {'name': 'gemini-2.5-pro', 'display_name': 'Gemini 2.5 Pro', 'type': 'pro'},
        ]
    
    @staticmethod
    def _classify_model(model_name: str) -> str:
        """Clasifica el modelo como 'flash' o 'pro'."""
        return 'pro' if 'pro' in model_name.lower() else 'flash'
    
    @staticmethod
    def _get_current_model() -> str:
        """Obtiene el modelo actual de la sesión."""
        try:
            import streamlit as st
            return st.session_state.get('selected_model', 'gemini-2.5-flash')
        except:
            return 'gemini-2.5-flash'
    
    @staticmethod
    def check_quota_available(
        model_name: str = None, 
        force_check: bool = False
    ) -> Tuple[bool, str]:
        """
        Verifica si la cuota está disponible usando caché.
        
        Args:
            model_name: Nombre del modelo a probar
            force_check: Si True, ignora caché y prueba nuevamente
            
        Returns:
            Tuple (available, status)
        """
        # Validar API key primero
        if not ServiceFactory.is_api_key_valid():
            return False, "API_KEY_NOT_CONFIGURED"
        
        # Usar caché si es válido
        if not force_check and ServiceFactory._is_api_cache_valid():
            logger.debug("Usando caché de estado de API")
            cache = ServiceFactory._api_status_cache
            return cache['available'], cache['status']
        
        # Obtener modelo a probar
        if model_name is None:
            model_name = ServiceFactory._get_current_model()
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            
            logger.info(f"Verificando cuota con modelo: {model_name}")
            
            # Probar con el modelo seleccionado
            test_model = genai.GenerativeModel(model_name)
            response = test_model.generate_content(
                "test",
                generation_config={'max_output_tokens': 5}
            )
            
            if response and response.text:
                ServiceFactory._update_api_cache(True, "OK")
                logger.info(f"Conexión exitosa con modelo: {model_name}")
                return True, "OK"
            
            ServiceFactory._update_api_cache(False, "API_ERROR")
            return False, "API_ERROR"
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.warning(f"Error verificando cuota: {error_msg[:150]}")
            
            # Clasificar error usando error_handler
            if APIErrorHandler.is_quota_error(error_msg):
                status = "QUOTA_EXCEEDED"
                available = False
            elif 'api_key' in error_msg or 'invalid' in error_msg or 'permission' in error_msg:
                status = "INVALID_API_KEY"
                available = False
            elif 'not found' in error_msg or '404' in error_msg:
                status = f"MODELO_NO_DISPONIBLE: {model_name}"
                available = False
            else:
                status = f"ERROR: {str(e)[:100]}"
                available = False
            
            ServiceFactory._update_api_cache(available, status)
            return available, status
    
    @staticmethod
    def create_rag_service(
        repo_name: str,
        repo_path: Path,
        repo_id: int,
        model_name: str = "gemini-2.5-flash",
        max_file_size_mb: int = 1,
        include_docs: bool = False
    ) -> Optional[RAGService]:
        """
        Crea un servicio RAG con el modelo especificado.
        
        Args:
            repo_name: Nombre del repositorio
            repo_path: Ruta física
            repo_id: ID en base de datos
            model_name: Modelo a usar
            max_file_size_mb: Tamaño máximo de archivo
            include_docs: Incluir documentación
            
        Returns:
            RAGService o None si error
        """
        # Verificar cuota antes de crear
        available, status = ServiceFactory.check_quota_available(model_name)
        
        if not available:
            logger.warning(f"No se puede crear RAGService: {status}")
            return None
        
        logger.info(f"Creando RAGService para {repo_name} con modelo {model_name}")
        
        try:
            return RAGService(
                repo_name=repo_name,
                repo_path=repo_path,
                repo_id=repo_id,
                model_name=model_name,
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
        
        Args:
            rag_service: Servicio RAG configurado
            
        Returns:
            AgentWorkflow o None si error
        """
        if rag_service is None:
            return None
        
        try:
            return AgentWorkflow(rag_service)
        except Exception as e:
            logger.error(f"Error creando AgentWorkflow: {e}")
            return None
    
    @staticmethod
    def setup_repository_services(
        repo: Repository,
        model_name: str = "gemini-2.5-flash",
        max_file_size_mb: int = 1,
        include_docs: bool = False
    ) -> Tuple[Optional[RAGService], Optional[AgentWorkflow]]:
        """
        Configura todos los servicios para un repositorio.
        
        Args:
            repo: Repositorio (debe tener db_id)
            model_name: Modelo a usar
            max_file_size_mb: Tamaño máximo de archivo
            include_docs: Incluir documentación
            
        Returns:
            Tupla (rag_service, agent_workflow)
        """
        rag_service = ServiceFactory.create_rag_service(
            repo_name=repo.name,
            repo_path=repo.path,
            repo_id=repo.db_id,
            model_name=model_name,
            max_file_size_mb=max_file_size_mb,
            include_docs=include_docs
        )
        
        agent_workflow = ServiceFactory.create_agent_workflow(rag_service) if rag_service else None
        
        return rag_service, agent_workflow