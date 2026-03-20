"""
Parser para CSS.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import re

from infrastructure.parsers.base_parser import BaseParser

logger = logging.getLogger(__name__)


class CSSParser(BaseParser):
    """
    Parser para archivos CSS.
    """
    
    def __init__(self):
        """Inicializa el parser de CSS."""
        super().__init__(
            language="css",
            extensions=['.css', '.scss', '.sass', '.less']
        )
        logger.info("Parser CSS inicializado")
    
    def can_parse(self, file_path: Path) -> bool:
        """
        Verifica si el archivo puede ser parseado.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            True si es archivo CSS
        """
        return file_path.suffix.lower() in self.extensions
    
    def parse_file(self, file_path: Path, content: str) -> Dict[str, Any]:
        """
        Parsea un archivo CSS.
        
        Args:
            file_path: Ruta del archivo
            content: Contenido del archivo
            
        Returns:
            Diccionario con selectores y reglas
        """
        return {
            'selectors': self.extract_selectors(content),
            'media_queries': self.extract_media_queries(content),
            'variables': self.extract_variables(content),
            'imports': self.extract_imports(content),
            'language': self.language
        }
    
    def extract_functions(self, content: str) -> List[Dict[str, Any]]:
        """
        CSS no tiene funciones, retorna lista vacía.
        """
        return []
    
    def extract_classes(self, content: str) -> List[Dict[str, Any]]:
        """
        Extrae clases CSS (selectores de clase).
        
        Args:
            content: Contenido del archivo
            
        Returns:
            Lista de clases CSS
        """
        classes = []
        
        # Buscar selectores de clase: .nombre-clase
        class_pattern = r'\.([a-zA-Z][a-zA-Z0-9_-]*)\s*\{'
        for match in re.finditer(class_pattern, content):
            classes.append({
                'name': match.group(1),
                'type': 'css-class',
                'line_start': content[:match.start()].count('\n') + 1
            })
        
        return classes
    
    def extract_imports(self, content: str) -> List[str]:
        """
        Extrae imports de otros archivos CSS.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            Lista de imports
        """
        imports = []
        
        # @import url('file.css')
        pattern = r'@import\s+(?:url\([\'"]?|[\'"]?)([^\'"\)]+)(?:[\'"]?\)?[\'"]?)\s*;'
        
        for match in re.finditer(pattern, content):
            imports.append(match.group(1).strip())
        
        return imports
    
    def extract_selectors(self, content: str) -> List[Dict[str, Any]]:
        """Extrae selectores CSS."""
        selectors = []
        
        # Eliminar comentarios
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        pattern = r'([^{]+)\{([^}]+)\}'
        
        for match in re.finditer(pattern, content):
            selector_text = match.group(1).strip()
            
            # Ignorar media queries y keyframes
            if selector_text.startswith('@'):
                continue
            
            for single_selector in selector_text.split(','):
                selector = single_selector.strip()
                if selector:
                    selectors.append({
                        'selector': selector,
                        'specificity': self._calculate_specificity(selector)
                    })
        
        return selectors
    
    def extract_media_queries(self, content: str) -> List[Dict[str, Any]]:
        """Extrae media queries."""
        media_queries = []
        
        pattern = r'@media\s+([^{]+)\{([^}]+)\}'
        for match in re.finditer(pattern, content, re.DOTALL):
            media_queries.append({
                'condition': match.group(1).strip(),
                'rules': match.group(2).strip()[:200] + '...' if len(match.group(2)) > 200 else match.group(2).strip()
            })
        
        return media_queries
    
    def extract_variables(self, content: str) -> List[Dict[str, str]]:
        """Extrae variables CSS."""
        variables = []
        
        # Variables CSS: --nombre: valor;
        pattern = r'--([a-zA-Z][a-zA-Z0-9_-]*)\s*:\s*([^;]+);'
        for match in re.finditer(pattern, content):
            variables.append({
                'name': f'--{match.group(1)}',
                'value': match.group(2).strip()
            })
        
        return variables
    
    def _calculate_specificity(self, selector: str) -> int:
        """Calcula la especificidad del selector."""
        id_count = len(re.findall(r'#[a-zA-Z][a-zA-Z0-9_-]*', selector))
        class_count = len(re.findall(r'\.[a-zA-Z][a-zA-Z0-9_-]*', selector))
        element_count = len(re.findall(r'\b[a-z][a-z0-9]*\b(?![#\.:])', selector, re.IGNORECASE))
        
        return (id_count * 100) + (class_count * 10) + element_count