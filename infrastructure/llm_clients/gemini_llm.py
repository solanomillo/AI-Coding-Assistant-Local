"""
Cliente para Google Gemini con modelo configurable.
Usa error_handler centralizado.
"""

import logging
import google.generativeai as genai
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

from infrastructure.llm_clients.error_handler import APIErrorHandler

load_dotenv()
logger = logging.getLogger(__name__)


class GeminiLLM:
    """
    Cliente simple para Gemini.
    Usa el modelo que el usuario selecciona.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "gemini-2.5-flash",
        auto_fallback: bool = False
    ):
        """
        Inicializa el cliente con el modelo especificado.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no encontrada")
        
        genai.configure(api_key=self.api_key)
        
        self.model_name = model
        self.auto_fallback = auto_fallback
        self.model = genai.GenerativeModel(model)
        self.chat_session = None
        self._quota_exceeded = False
        
        logger.info(f"GeminiLLM inicializado con modelo: {model}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        retry_count: int = 2
    ) -> str:
        """
        Genera texto con el modelo configurado.
        """
        if self._quota_exceeded:
            return APIErrorHandler.ERROR_MESSAGES['QUOTA_EXCEEDED']
        
        last_error = None
        
        for attempt in range(retry_count + 1):
            try:
                generation_config = {'temperature': temperature}
                if max_tokens:
                    generation_config['max_output_tokens'] = max_tokens
                
                response = self.model.generate_content(prompt, generation_config=generation_config)
                return response.text
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                logger.warning(f"Intento {attempt + 1} falló: {error_msg[:200]}")
                
                if APIErrorHandler.is_quota_error(error_msg):
                    self._quota_exceeded = True
                    return APIErrorHandler.ERROR_MESSAGES['QUOTA_EXCEEDED']
                
                if APIErrorHandler.is_rate_limit_error(error_msg):
                    APIErrorHandler.handle_rate_limit(error_msg, attempt)
                    continue
                
                if self.auto_fallback and attempt < retry_count:
                    if self._switch_model():
                        continue
        
        error_msg = str(last_error) if last_error else "Error desconocido"
        
        if APIErrorHandler.is_quota_error(error_msg):
            self._quota_exceeded = True
            return APIErrorHandler.ERROR_MESSAGES['QUOTA_EXCEEDED']
        
        return APIErrorHandler.get_friendly_message(error_msg)
    
    def _switch_model(self) -> bool:
        """Intenta cambiar a un modelo alternativo."""
        fallback_models = ["gemini-2.0-flash", "gemini-1.5-flash"]
        
        for model_name in fallback_models:
            if model_name != self.model_name:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    self.model_name = model_name
                    logger.info(f"Cambiado a modelo: {model_name}")
                    return True
                except Exception:
                    continue
        
        return False
    
    def start_chat(self) -> None:
        """Inicia sesión de chat."""
        if self._quota_exceeded:
            logger.warning("No se puede iniciar chat: cuota agotada")
            return
        
        self.chat_session = self.model.start_chat()
        logger.info("Sesión de chat iniciada")
    
    def chat(self, message: str) -> str:
        """Envía mensaje en chat."""
        if self._quota_exceeded:
            return APIErrorHandler.ERROR_MESSAGES['QUOTA_EXCEEDED']
        
        if not self.chat_session:
            self.start_chat()
        
        try:
            response = self.chat_session.send_message(message)
            return response.text
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error en chat: {error_msg[:200]}")
            
            if APIErrorHandler.is_quota_error(error_msg):
                self._quota_exceeded = True
                return APIErrorHandler.ERROR_MESSAGES['QUOTA_EXCEEDED']
            
            if self.auto_fallback:
                logger.info("Reiniciando sesión de chat...")
                self.chat_session = None
                self.start_chat()
                try:
                    response = self.chat_session.send_message(message)
                    return response.text
                except Exception as e2:
                    logger.error(f"Error en reintento: {e2}")
                    return APIErrorHandler.get_friendly_message(str(e2))
            
            return APIErrorHandler.get_friendly_message(error_msg)
    
    def reset_chat(self) -> None:
        """Reinicia sesión de chat."""
        self.chat_session = None
        logger.info("Chat reiniciado")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Información del modelo actual."""
        return {
            'current_model': self.model_name,
            'auto_fallback': self.auto_fallback,
            'quota_exceeded': self._quota_exceeded
        }
    
    def reset_quota_flag(self) -> None:
        """Reinicia la bandera de cuota agotada."""
        self._quota_exceeded = False
        logger.info("Bandera de cuota reiniciada")