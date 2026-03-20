"""
Parser para HTML.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import re

from infrastructure.parsers.base_parser import BaseParser

logger = logging.getLogger(__name__)


class HTMLParser(BaseParser):
    """
    Parser para archivos HTML.
    """
    
    def __init__(self):
        """Inicializa el parser de HTML."""
        super().__init__(
            language="html",
            extensions=['.html', '.htm']
        )
        logger.info("Parser HTML inicializado")
    
    def can_parse(self, file_path: Path) -> bool:
        """
        Verifica si el archivo puede ser parseado.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            True si es archivo HTML
        """
        return file_path.suffix.lower() in self.extensions
    
    def parse_file(self, file_path: Path, content: str) -> Dict[str, Any]:
        """
        Parsea un archivo HTML.
        
        Args:
            file_path: Ruta del archivo
            content: Contenido del archivo
            
        Returns:
            Diccionario con estructura del HTML
        """
        return {
            'title': self.extract_title(content),
            'scripts': self.extract_scripts(content),
            'styles': self.extract_styles(content),
            'links': self.extract_links(content),
            'images': self.extract_images(content),
            'forms': self.extract_forms(content),
            'language': self.language
        }
    
    def extract_functions(self, content: str) -> List[Dict[str, Any]]:
        """
        HTML no tiene funciones, retorna lista vacía.
        """
        return []
    
    def extract_classes(self, content: str) -> List[Dict[str, Any]]:
        """
        HTML no tiene clases, retorna lista vacía.
        """
        return []
    
    def extract_imports(self, content: str) -> List[str]:
        """
        Extrae imports de scripts y estilos.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            Lista de imports
        """
        imports = []
        
        # Scripts externos
        script_pattern = r'<script[^>]*src=["\'](.*?)["\'][^>]*>'
        for match in re.finditer(script_pattern, content, re.IGNORECASE):
            imports.append(f"script: {match.group(1)}")
        
        # CSS externos
        css_pattern = r'<link[^>]*href=["\'](.*?\.css[^"\']*)["\'][^>]*>'
        for match in re.finditer(css_pattern, content, re.IGNORECASE):
            imports.append(f"css: {match.group(1)}")
        
        return imports
    
    def extract_title(self, content: str) -> str:
        """Extrae el título de la página."""
        match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ''
    
    def extract_scripts(self, content: str) -> List[Dict[str, Any]]:
        """Extrae scripts inline y externos."""
        scripts = []
        
        # Scripts externos
        src_pattern = r'<script[^>]*src=["\'](.*?)["\'][^>]*>'
        for match in re.finditer(src_pattern, content, re.IGNORECASE):
            scripts.append({
                'type': 'external',
                'src': match.group(1)
            })
        
        # Scripts inline
        inline_pattern = r'<script[^>]*>(.*?)</script>'
        for match in re.finditer(inline_pattern, content, re.IGNORECASE | re.DOTALL):
            script_content = match.group(1).strip()
            if script_content:
                scripts.append({
                    'type': 'inline',
                    'content': script_content[:200] + '...' if len(script_content) > 200 else script_content
                })
        
        return scripts
    
    def extract_styles(self, content: str) -> List[Dict[str, Any]]:
        """Extrae estilos CSS."""
        styles = []
        
        # CSS externos
        link_pattern = r'<link[^>]*href=["\'](.*?\.css[^"\']*)["\'][^>]*>'
        for match in re.finditer(link_pattern, content, re.IGNORECASE):
            styles.append({
                'type': 'external',
                'href': match.group(1)
            })
        
        # CSS inline
        style_pattern = r'<style[^>]*>(.*?)</style>'
        for match in re.finditer(style_pattern, content, re.IGNORECASE | re.DOTALL):
            style_content = match.group(1).strip()
            if style_content:
                styles.append({
                    'type': 'inline',
                    'content': style_content[:200] + '...' if len(style_content) > 200 else style_content
                })
        
        return styles
    
    def extract_links(self, content: str) -> List[Dict[str, str]]:
        """Extrae enlaces del documento."""
        links = []
        pattern = r'<a[^>]*href=["\'](.*?)["\'][^>]*>(.*?)</a>'
        
        for match in re.finditer(pattern, content, re.IGNORECASE | re.DOTALL):
            links.append({
                'href': match.group(1),
                'text': re.sub(r'<[^>]+>', '', match.group(2)).strip()
            })
        
        return links
    
    def extract_images(self, content: str) -> List[Dict[str, str]]:
        """Extrae imágenes del documento."""
        images = []
        pattern = r'<img[^>]*src=["\'](.*?)["\'][^>]*>'
        
        for match in re.finditer(pattern, content, re.IGNORECASE):
            images.append({
                'src': match.group(1),
                'alt': self._extract_attr(match.group(0), 'alt')
            })
        
        return images
    
    def extract_forms(self, content: str) -> List[Dict[str, Any]]:
        """Extrae formularios."""
        forms = []
        pattern = r'<form[^>]*>(.*?)</form>'
        
        for match in re.finditer(pattern, content, re.IGNORECASE | re.DOTALL):
            forms.append({
                'method': self._extract_attr(match.group(0), 'method'),
                'action': self._extract_attr(match.group(0), 'action'),
                'inputs': len(re.findall(r'<input', match.group(1), re.IGNORECASE))
            })
        
        return forms
    
    def _extract_attr(self, tag: str, attr: str) -> str:
        """Extrae un atributo de una etiqueta HTML."""
        pattern = f'{attr}=["\'](.*?)["\']'
        match = re.search(pattern, tag, re.IGNORECASE)
        return match.group(1) if match else ''