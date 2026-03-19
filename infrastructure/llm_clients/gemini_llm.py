"""
Cliente para Google Gemini con soporte para versiones gratuita y pro.

Este módulo maneja la generación de respuestas usando
los modelos de texto de Gemini con detección automática
de disponibilidad y fallback.
"""

import logging
import google.generativeai as genai
from typing import Optional, List, Dict, Any, Tuple
import os
from dotenv import load_dotenv
import time

load_dotenv()
logger = logging.getLogger(__name__)


class GeminiLLM:
    """
    Cliente para Gemini con soporte multi-versión.
    
    Características:
    - Soporte para modelos gratuitos (flash) y pro
    - Detección automática de disponibilidad
    - Fallback entre modelos si uno falla
    - Manejo de rate limits
    """
    
    # Modelos disponibles ordenados por preferencia
    AVAILABLE_MODELS = {
        'pro': [
            "gemini-2.5-pro-exp-03-25",  # Pro experimental (más capaz)
            "gemini-2.0-pro-exp-02-05",   # Pro anterior
        ],
        'flash': [
            "gemini-2.5-flash",           # Flash rápido (gratuito)
            "gemini-2.0-flash-exp",        # Flash experimental
            "gemini-1.5-flash",            # Flash versión anterior
        ],
        'free': [
            "gemini-2.5-flash",            # El que mejor funciona gratis
            "gemini-1.5-flash-8b",         # Modelo pequeño y rápido
        ]
    }
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: Optional[str] = None,
        prefer_pro: bool = False,
        auto_fallback: bool = True
    ):
        """
        Inicializa el cliente Gemini con soporte multi-modelo.
        
        Args:
            api_key: API key de Gemini
            model: Modelo específico a usar (opcional)
            prefer_pro: Si True, intenta usar modelos pro primero
            auto_fallback: Si True, intenta otros modelos si falla
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no encontrada en .env")
        
        # Configurar Gemini
        genai.configure(api_key=self.api_key)
        
        self.prefer_pro = prefer_pro
        self.auto_fallback = auto_fallback
        self.current_model = None
        self.model_instance = None
        self.chat_session = None
        self.last_error = None
        
        # Inicializar modelo
        if model:
            # Usar modelo específico si se proporciona
            self.current_model = model
            self.model_instance = self._create_model(model)
            logger.info(f"✅ GeminiLLM inicializado con modelo específico: {model}")
        else:
            # Detectar mejor modelo disponible
            self._initialize_best_model()
    
    def _create_model(self, model_name: str) -> Optional[genai.GenerativeModel]:
        """
        Crea una instancia del modelo.
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            Instancia del modelo o None si falla
        """
        try:
            return genai.GenerativeModel(model_name)
        except Exception as e:
            logger.warning(f"⚠️ No se pudo crear modelo {model_name}: {e}")
            return None
    
    def _test_model(self, model_name: str) -> bool:
        """
        Prueba si un modelo está disponible y funciona.
        
        Args:
            model_name: Nombre del modelo a probar
            
        Returns:
            True si el modelo funciona
        """
        try:
            model = genai.GenerativeModel(model_name)
            # Prueba simple
            response = model.generate_content(
                "test",
                generation_config={'max_output_tokens': 5}
            )
            return bool(response and response.text)
        except Exception as e:
            logger.debug(f"Modelo {model_name} no disponible: {e}")
            return False
    
    def _initialize_best_model(self) -> None:
        """
        Inicializa el mejor modelo disponible según preferencias.
        """
        # Determinar orden de modelos a probar
        models_to_try = []
        
        if self.prefer_pro:
            models_to_try.extend(self.AVAILABLE_MODELS['pro'])
            models_to_try.extend(self.AVAILABLE_MODELS['flash'])
        else:
            models_to_try.extend(self.AVAILABLE_MODELS['flash'])
            models_to_try.extend(self.AVAILABLE_MODELS['pro'])
        
        # Siempre incluir modelos gratuitos al final
        models_to_try.extend(self.AVAILABLE_MODELS['free'])
        
        # Eliminar duplicados manteniendo orden
        seen = set()
        unique_models = []
        for model in models_to_try:
            if model not in seen:
                seen.add(model)
                unique_models.append(model)
        
        # Probar modelos en orden
        for model_name in unique_models:
            logger.info(f"Probando modelo: {model_name}")
            if self._test_model(model_name):
                self.current_model = model_name
                self.model_instance = self._create_model(model_name)
                logger.info(f"✅ Usando modelo: {model_name}")
                return
        
        # Si ningún modelo funciona, lanzar error
        raise RuntimeError(
            "No se pudo inicializar ningún modelo Gemini. "
            "Verifica tu API key y conexión."
        )
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        retry_count: int = 2
    ) -> str:
        """
        Genera texto con Gemini, con fallback automático.
        
        Args:
            prompt: Prompt de entrada
            temperature: Temperatura (0.0 = determinista, 1.0 = creativo)
            max_tokens: Máximo de tokens a generar
            retry_count: Número de reintentos en caso de error
            
        Returns:
            Texto generado
        """
        last_error = None
        
        for attempt in range(retry_count + 1):
            try:
                # Configurar generación
                generation_config = {
                    'temperature': temperature,
                }
                if max_tokens:
                    generation_config['max_output_tokens'] = max_tokens
                
                # Generar
                response = self.model_instance.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                return response.text
                
            except Exception as e:
                last_error = e
                logger.warning(f"Intento {attempt + 1} falló: {e}")
                
                # Si es error de rate limit, esperar
                if "rate limit" in str(e).lower():
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Rate limit, esperando {wait_time}s...")
                    time.sleep(wait_time)
                
                # Si auto_fallback está activado, intentar cambiar modelo
                if self.auto_fallback and attempt < retry_count:
                    if self._switch_to_fallback_model():
                        logger.info("Cambiado a modelo de respaldo")
                        continue
        
        # Si llegamos aquí, todos los intentos fallaron
        error_msg = f"Error generando respuesta después de {retry_count} intentos: {last_error}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
    
    def _switch_to_fallback_model(self) -> bool:
        """
        Cambia a un modelo de respaldo.
        
        Returns:
            True si se pudo cambiar a otro modelo
        """
        current = self.current_model
        all_models = (
            self.AVAILABLE_MODELS['pro'] + 
            self.AVAILABLE_MODELS['flash'] + 
            self.AVAILABLE_MODELS['free']
        )
        
        # Encontrar siguiente modelo disponible
        for model in all_models:
            if model != current and self._test_model(model):
                self.current_model = model
                self.model_instance = self._create_model(model)
                logger.info(f"✅ Cambiado a modelo: {model}")
                return True
        
        return False
    
    def start_chat(self) -> None:
        """Inicia sesión de chat."""
        if not self.model_instance:
            raise RuntimeError("Modelo no inicializado")
        
        self.chat_session = self.model_instance.start_chat()
        logger.info("Sesión de chat iniciada")
    
    def chat(self, message: str) -> str:
        """
        Envía mensaje en chat.
        
        Args:
            message: Mensaje del usuario
            
        Returns:
            Respuesta del chat
        """
        if not self.chat_session:
            self.start_chat()
        
        try:
            response = self.chat_session.send_message(message)
            return response.text
        except Exception as e:
            logger.error(f"Error en chat: {e}")
            
            # Si el chat falla, intentar reiniciar
            if self.auto_fallback:
                logger.info("Reiniciando sesión de chat...")
                self.chat_session = None
                self.start_chat()
                try:
                    response = self.chat_session.send_message(message)
                    return response.text
                except:
                    pass
            
            return f"Error en chat: {str(e)}"
    
    def reset_chat(self) -> None:
        """Reinicia sesión de chat."""
        self.chat_session = None
        logger.info("Chat reiniciado")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el modelo actual.
        
        Returns:
            Diccionario con información del modelo
        """
        model_type = "pro" if self.current_model in self.AVAILABLE_MODELS['pro'] else "flash"
        
        return {
            'current_model': self.current_model,
            'model_type': model_type,
            'prefer_pro': self.prefer_pro,
            'auto_fallback': self.auto_fallback,
            'available_models': {
                'pro': self.AVAILABLE_MODELS['pro'],
                'flash': self.AVAILABLE_MODELS['flash']
            }
        }
    
    def list_available_models(self) -> List[str]:
        """
        Lista todos los modelos disponibles (que responden).
        
        Returns:
            Lista de modelos que funcionan
        """
        available = []
        all_models = (
            self.AVAILABLE_MODELS['pro'] + 
            self.AVAILABLE_MODELS['flash'] + 
            self.AVAILABLE_MODELS['free']
        )
        
        for model in set(all_models):  # Usar set para eliminar duplicados
            if self._test_model(model):
                available.append(model)
        
        return available


# Ejemplo de uso:
if __name__ == "__main__":
    # Probar diferentes configuraciones
    print("🔧 Probando GeminiLLM...")
    
    # Configuración 1: Gratuita (flash)
    llm_free = GeminiLLM(prefer_pro=False)
    print(f"Modelo gratuito: {llm_free.get_model_info()}")
    
    # Configuración 2: Preferir Pro si disponible
    llm_pro = GeminiLLM(prefer_pro=True)
    print(f"Preferencia Pro: {llm_pro.get_model_info()}")
    
    # Probar generación
    response = llm_free.generate("Hola, ¿cómo estás?", max_tokens=50)
    print(f"Respuesta: {response[:100]}...")