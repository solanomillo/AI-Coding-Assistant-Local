"""
Fábrica de servicios para centralizar la creación de componentes.
Optimizada con caché agresivo para ahorrar cuota.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from application.services.rag_gemini_service import RAGService
from application.graph.workflow import AgentWorkflow
from domain.models.repository import Repository

logger = logging.getLogger(__name__)


class ServiceFactory:
    """
    Fábrica para crear servicios de manera centralizada.
    """
    
    # Cache de estado de API (para no probar constantemente)
    _api_status_cache = {
        'available': None,
        'status': None,
        'last_check': 0,
        'cache_duration_ok': 300,      # 5 minutos si está OK
        'cache_duration_exhausted': 3600  # 1 hora si está agotada
    }
    
    # Cache de modelos disponibles
    _available_models_cache = None
    _models_cache_timestamp = 0
    _models_cache_duration = 3600  # 1 hora
    
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
            'last_check': 0,
            'cache_duration_ok': 300,
            'cache_duration_exhausted': 3600
        }
        ServiceFactory._available_models_cache = None
        ServiceFactory._models_cache_timestamp = 0
        logger.info("Cachés limpiados")
    
    @staticmethod
    def _is_cache_valid() -> bool:
        """Verifica si el caché de estado de API es válido."""
        if ServiceFactory._api_status_cache['available'] is None:
            return False
        
        now = time.time()
        cache_age = now - ServiceFactory._api_status_cache['last_check']
        
        if ServiceFactory._api_status_cache['available']:
            # Si está OK, caché dura 5 minutos
            return cache_age < ServiceFactory._api_status_cache['cache_duration_ok']
        else:
            # Si está agotada o error, caché dura 1 hora
            return cache_age < ServiceFactory._api_status_cache['cache_duration_exhausted']
    
    @staticmethod
    def get_available_models(force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Obtiene la lista de modelos disponibles."""
        now = time.time()
        
        if not force_refresh and ServiceFactory._available_models_cache is not None:
            cache_age = now - ServiceFactory._models_cache_timestamp
            if cache_age < ServiceFactory._models_cache_duration:
                logger.debug(f"Usando caché de modelos (edad: {cache_age:.1f}s)")
                return ServiceFactory._available_models_cache
        
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
            
            logger.info(f"Modelos obtenidos de API: {len(models)}")
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
        name_lower = model_name.lower()
        if 'pro' in name_lower:
            return 'pro'
        return 'flash'
    
    @staticmethod
    def check_quota_available(model_name: str = None, force_check: bool = False) -> Tuple[bool, str]:
        """
        Verifica si la cuota está disponible usando caché.
        
        Args:
            model_name: Nombre del modelo a probar
            force_check: Si True, ignora caché y prueba nuevamente
        """
        if not ServiceFactory.is_api_key_valid():
            return False, "API_KEY_NOT_CONFIGURED"
        
        # Usar caché si es válido y no se fuerza verificación
        if not force_check and ServiceFactory._is_cache_valid():
            logger.debug("Usando caché de estado de API")
            return ServiceFactory._api_status_cache['available'], ServiceFactory._api_status_cache['status']
        
        # Si no se especifica modelo, usar el de sesión o el por defecto
        if model_name is None:
            try:
                import streamlit as st
                model_name = st.session_state.get('selected_model', 'gemini-2.5-flash')
            except:
                model_name = 'gemini-2.5-flash'
        
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
                # Actualizar caché con estado OK
                ServiceFactory._api_status_cache['available'] = True
                ServiceFactory._api_status_cache['status'] = "OK"
                ServiceFactory._api_status_cache['last_check'] = time.time()
                logger.info(f"Conexión exitosa con modelo: {model_name}")
                return True, "OK"
            
            return False, "API_ERROR"
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.warning(f"Error verificando cuota: {error_msg[:150]}")
            
            # Clasificar error
            if 'quota' in error_msg or '429' in error_msg or 'rate limit' in error_msg:
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
            
            # Actualizar caché
            ServiceFactory._api_status_cache['available'] = available
            ServiceFactory._api_status_cache['status'] = status
            ServiceFactory._api_status_cache['last_check'] = time.time()
            
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
        """Crea un servicio RAG con el modelo especificado."""
        # Verificar cuota antes de crear (usa caché)
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
        """Crea un flujo de agentes a partir de un servicio RAG."""
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
        """Configura todos los servicios para un repositorio."""
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