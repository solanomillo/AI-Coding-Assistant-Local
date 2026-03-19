"""
Parser para CSS.
Extrae selectores, reglas y variables.
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
    Extrae selectores, propiedades y variables.
    """
    
    def __init__(self):
        """Inicializa el parser de CSS."""
        super().__init__(
            language="css",
            extensions=['.css', '.scss', '.sass', '.less']
        )
        logger.info("Parser CSS inicializado")
    
    def parse_file(self, file_path: Path, content: str) -> Dict[str, Any]:
        """
        Parsea un archivo CSS.
        
        Args:
            file_path: Ruta del archivo
            content: Contenido del archivo
            
        Returns:
            Diccionario con selectores y reglas
        """
        result = {
            'selectors': self._extract_selectors(content),
            'media_queries': self._extract_media_queries(content),
            'keyframes': self._extract_keyframes(content),
            'variables': self._extract_variables(content),
            'imports': self._extract_imports(content),
            'line_count': len(content.splitlines())
        }
        
        logger.debug(f"CSS parseado: {len(result['selectors'])} selectores")
        
        return result
    
    def _extract_selectors(self, content: str) -> List[Dict[str, Any]]:
        """Extrae selectores CSS con sus propiedades."""
        selectors = []
        
        # Eliminar comentarios
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Patrón para selector + reglas
        pattern = r'([^{]+)\{([^}]+)\}'
        
        for match in re.finditer(pattern, content):
            selector_text = match.group(1).strip()
            rules_text = match.group(2).strip()
            
            # Ignorar media queries y keyframes
            if selector_text.startswith('@'):
                continue
            
            # Múltiples selectores separados por coma
            for single_selector in selector_text.split(','):
                selector = single_selector.strip()
                if selector:
                    selectors.append({
                        'selector': selector,
                        'properties': self._extract_properties(rules_text),
                        'specificity': self._calculate_specificity(selector),
                        'line_start': content[:match.start()].count('\n') + 1
                    })
        
        return selectors
    
    def _extract_properties(self, rules: str) -> List[Dict[str, str]]:
        """Extrae propiedades CSS de una regla."""
        properties = []
        
        # Separar por punto y coma, ignorando los que están dentro de valores
        parts = re.split(r';(?![^(]*\))', rules)
        
        for part in parts:
            if ':' in part:
                prop, value = part.split(':', 1)
                properties.append({
                    'property': prop.strip(),
                    'value': value.strip()
                })
        
        return properties
    
    def _extract_media_queries(self, content: str) -> List[Dict[str, Any]]:
        """Extrae media queries."""
        media_queries = []
        
        pattern = r'@media\s+([^{]+)\{([^}]+)\}'
        for match in re.finditer(pattern, content, re.DOTALL):
            media_queries.append({
                'condition': match.group(1).strip(),
                'rules': match.group(2).strip()[:200] + '...' if len(match.group(2)) > 200 else match.group(2).strip(),
                'line_start': content[:match.start()].count('\n') + 1
            })
        
        return media_queries
    
    def _extract_keyframes(self, content: str) -> List[Dict[str, Any]]:
        """Extrae keyframes de animación."""
        keyframes = []
        
        pattern = r'@keyframes\s+([^{]+)\{([^}]+)\}'
        for match in re.finditer(pattern, content, re.DOTALL):
            keyframes.append({
                'name': match.group(1).strip(),
                'content': match.group(2).strip()[:200] + '...' if len(match.group(2)) > 200 else match.group(2).strip(),
                'line_start': content[:match.start()].count('\n') + 1
            })
        
        return keyframes
    
    def _extract_variables(self, content: str) -> List[Dict[str, str]]:
        """Extrae variables CSS (custom properties)."""
        variables = []
        
        # Variables CSS nativas: --nombre: valor;
        pattern = r'--([a-zA-Z][a-zA-Z0-9_-]*)\s*:\s*([^;]+);'
        
        for match in re.finditer(pattern, content):
            variables.append({
                'name': match.group(1),
                'value': match.group(2).strip(),
                'line_start': content[:match.start()].count('\n') + 1
            })
        
        # Variables SCSS: $nombre: valor;
        pattern = r'\$([a-zA-Z][a-zA-Z0-9_-]*)\s*:\s*([^;]+);'
        
        for match in re.finditer(pattern, content):
            variables.append({
                'name': f'${match.group(1)}',
                'value': match.group(2).strip(),
                'line_start': content[:match.start()].count('\n') + 1
            })
        
        return variables
    
    def _extract_imports(self, content: str) -> List[str]:
        """Extrae imports de otros archivos."""
        imports = []
        
        # @import url('file.css')
        pattern = r'@import\s+(?:url\([\'"]?|[\'"]?)([^\'"\)]+)(?:[\'"]?\)?[\'"]?)\s*;'
        
        for match in re.finditer(pattern, content):
            imports.append(match.group(1).strip())
        
        return imports
    
    def _calculate_specificity(self, selector: str) -> int:
        """
        Calcula la especificidad del selector.
        Mayor número = más específico.
        """
        # IDs
        id_count = len(re.findall(r'#[a-zA-Z][a-zA-Z0-9_-]*', selector))
        
        # Clases y pseudo-clases
        class_count = len(re.findall(r'\.[a-zA-Z][a-zA-Z0-9_-]*', selector))
        class_count += len(re.findall(r':[a-z-]+', selector))
        
        # Elementos
        element_pattern = r'\b[a-z][a-z0-9]*\b(?![#\.:])'
        element_count = len(re.findall(element_pattern, selector, re.IGNORECASE))
        
        # Cálculo: IDs * 100 + clases * 10 + elementos
        return (id_count * 100) + (class_count * 10) + element_count