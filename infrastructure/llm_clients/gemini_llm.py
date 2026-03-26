"""
Cliente para Google Gemini con modelo configurable.
Sin pruebas automáticas, solo usa el modelo que el usuario selecciona.
"""

import logging
import google.generativeai as genai
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class GeminiLLM:
    """
    Cliente simple para Gemini.
    Usa el modelo que el usuario selecciona, sin pruebas automáticas.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "gemini-2.5-flash",
        auto_fallback: bool = False
    ):
        """
        Inicializa el cliente con el modelo especificado.
        
        Args:
            api_key: API key de Gemini
            model: Modelo a usar (ej: "gemini-2.5-flash", "gemini-2.5-pro")
            auto_fallback: Si True, intenta cambiar de modelo en caso de error
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no encontrada")
        
        genai.configure(api_key=self.api_key)
        
        self.model_name = model
        self.auto_fallback = auto_fallback
        self.model = genai.GenerativeModel(model)
        self.chat_session = None
        
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
                logger.warning(f"Intento {attempt + 1} falló: {str(e)[:100]}")
                
                if self.auto_fallback and attempt < retry_count:
                    if self._switch_model():
                        continue
        
        return f"Error: {last_error}"
    
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
        self.chat_session = self.model.start_chat()
        logger.info("Sesión de chat iniciada")
    
    def chat(self, message: str) -> str:
        """Envía mensaje en chat."""
        if not self.chat_session:
            self.start_chat()
        
        try:
            response = self.chat_session.send_message(message)
            return response.text
        except Exception as e:
            logger.error(f"Error en chat: {e}")
            return f"Error: {e}"
    
    def reset_chat(self) -> None:
        """Reinicia sesión de chat."""
        self.chat_session = None
        logger.info("Chat reiniciado")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Información del modelo actual."""
        return {
            'current_model': self.model_name,
            'auto_fallback': self.auto_fallback
        }