"""
Parser base abstracto para todos los lenguajes.
Define la interfaz común que deben implementar todos los parsers.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """
    Clase base abstracta para parsers de lenguajes de programación.
    Todos los parsers específicos deben heredar de esta clase.
    """
    
    def __init__(self, language: str, extensions: List[str]):
        """
        Inicializa el parser base.
        
        Args:
            language: Nombre del lenguaje
            extensions: Lista de extensiones soportadas
        """
        self.language = language
        self.extensions = [ext.lower() for ext in extensions]
        logger.info(f"Parser base inicializado para {language}")
    
    def can_parse(self, file_path: Path) -> bool:
        """
        Verifica si el archivo puede ser parseado por este parser.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            True si el archivo tiene extensión soportada
        """
        return file_path.suffix.lower() in self.extensions
    
    @abstractmethod
    def parse_file(self, file_path: Path, content: str) -> Dict[str, Any]:
        """
        Parsea un archivo y extrae su información.
        
        Args:
            file_path: Ruta del archivo
            content: Contenido del archivo
            
        Returns:
            Diccionario con la información extraída
        """
        pass
    
    def extract_functions(self, content: str) -> List[Dict[str, Any]]:
        """
        Extrae funciones del contenido.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            Lista de funciones encontradas
        """
        return []
    
    def extract_classes(self, content: str) -> List[Dict[str, Any]]:
        """
        Extrae clases del contenido.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            Lista de clases encontradas
        """
        return []
    
    def extract_imports(self, content: str) -> List[str]:
        """
        Extrae importaciones/dependencias.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            Lista de dependencias
        """
        return []
    
    def get_language_info(self) -> Dict[str, Any]:
        """
        Obtiene información del parser.
        
        Returns:
            Diccionario con información del lenguaje
        """
        return {
            'language': self.language,
            'extensions': self.extensions,
            'parser_class': self.__class__.__name__
        }