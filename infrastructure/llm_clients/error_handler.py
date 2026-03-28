"""
Manejador centralizado de errores de API.
"""

import logging
import time
import re
from typing import Optional

logger = logging.getLogger(__name__)


class APIErrorHandler:
    """
    Manejador centralizado de errores de API.
    """
    
    # Mensajes amigables para diferentes tipos de error
    ERROR_MESSAGES = {
        'QUOTA_EXCEEDED': "⚠️ **Límite diario de API alcanzado.** Las consultas estarán disponibles mañana. Por favor, intenta nuevamente más tarde.",
        'RATE_LIMIT': "⚠️ **Demasiadas solicitudes.** Por favor, espera unos segundos y vuelve a intentarlo.",
        'API_KEY_INVALID': "❌ **API Key inválida.** Verifica que la hayas copiado correctamente.",
        'API_KEY_MISSING': "🔑 **API Key no configurada.** Ve a Configuración para agregarla.",
        'MODEL_NOT_FOUND': "❌ **Modelo no disponible.** El modelo seleccionado no está disponible con tu API key.",
        'NETWORK_ERROR': "🌐 **Error de red.** Verifica tu conexión a internet.",
        'UNKNOWN': "❌ **Error desconocido.** Por favor, intenta nuevamente más tarde."
    }
    
    @staticmethod
    def is_quota_error(error_msg: str) -> bool:
        """
        Detecta si el error es por cuota agotada.
        
        Args:
            error_msg: Mensaje de error
            
        Returns:
            True si es error de cuota
        """
        error_lower = error_msg.lower()
        return ('429' in error_lower or 
                'quota' in error_lower or 
                'rate limit' in error_lower or
                'exceeded' in error_lower)
    
    @staticmethod
    def is_rate_limit_error(error_msg: str) -> bool:
        """
        Detecta si el error es por rate limit.
        
        Args:
            error_msg: Mensaje de error
            
        Returns:
            True si es error de rate limit
        """
        error_lower = error_msg.lower()
        return ('429' in error_lower or 
                'rate limit' in error_lower or
                'too many requests' in error_lower)
    
    @staticmethod
    def get_friendly_message(error_msg: str) -> str:
        """
        Obtiene un mensaje amigable para el error.
        
        Args:
            error_msg: Mensaje de error original
            
        Returns:
            Mensaje amigable para el usuario
        """
        error_lower = error_msg.lower()
        
        if APIErrorHandler.is_quota_error(error_msg):
            return APIErrorHandler.ERROR_MESSAGES['QUOTA_EXCEEDED']
        elif 'api_key' in error_lower and ('invalid' in error_lower or 'not found' in error_lower):
            return APIErrorHandler.ERROR_MESSAGES['API_KEY_INVALID']
        elif 'network' in error_lower or 'connection' in error_lower or 'timeout' in error_lower:
            return APIErrorHandler.ERROR_MESSAGES['NETWORK_ERROR']
        elif 'not found' in error_lower or '404' in error_lower:
            return APIErrorHandler.ERROR_MESSAGES['MODEL_NOT_FOUND']
        
        return APIErrorHandler.ERROR_MESSAGES['UNKNOWN']
    
    @staticmethod
    def extract_retry_delay(error_msg: str) -> Optional[int]:
        """
        Extrae el tiempo de espera recomendado del error.
        
        Args:
            error_msg: Mensaje de error
            
        Returns:
            Tiempo de espera en segundos o None
        """
        match = re.search(r'retry in (\d+(?:\.\d+)?)', error_msg.lower())
        if match:
            return int(float(match.group(1))) + 1
        return None
    
    @staticmethod
    def handle_rate_limit(error_msg: str, attempt: int) -> None:
        """
        Maneja un error de rate limit con backoff exponencial.
        
        Args:
            error_msg: Mensaje de error
            attempt: Número de intento actual
        """
        wait_time = APIErrorHandler.extract_retry_delay(error_msg)
        if not wait_time:
            wait_time = 2 ** attempt
        
        logger.info(f"Rate limit detectado, esperando {wait_time} segundos...")
        time.sleep(wait_time)