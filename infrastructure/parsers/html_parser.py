"""
Parser para HTML.
Extrae estructura, scripts y estilos.
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
    Extrae scripts, estilos y estructura básica.
    """
    
    def __init__(self):
        """Inicializa el parser de HTML."""
        super().__init__(
            language="html",
            extensions=['.html', '.htm']
        )
        logger.info("Parser HTML inicializado")
    
    def parse_file(self, file_path: Path, content: str) -> Dict[str, Any]:
        """
        Parsea un archivo HTML.
        
        Args:
            file_path: Ruta del archivo
            content: Contenido del archivo
            
        Returns:
            Diccionario con estructura del HTML
        """
        result = {
            'title': self._extract_title(content),
            'scripts': self._extract_scripts(content),
            'styles': self._extract_styles(content),
            'links': self._extract_links(content),
            'images': self._extract_images(content),
            'forms': self._extract_forms(content),
            'meta_tags': self._extract_meta_tags(content),
            'line_count': len(content.splitlines())
        }
        
        logger.debug(f"HTML parseado: {len(result['scripts'])} scripts, "
                    f"{len(result['styles'])} estilos")
        
        return result
    
    def _extract_title(self, content: str) -> str:
        """Extrae el título de la página."""
        match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ''
    
    def _extract_scripts(self, content: str) -> List[Dict[str, Any]]:
        """Extrae scripts inline y externos."""
        scripts = []
        
        # Scripts inline
        inline_pattern = r'<script[^>]*>(.*?)</script>'
        for match in re.finditer(inline_pattern, content, re.IGNORECASE | re.DOTALL):
            script_content = match.group(1).strip()
            if script_content:
                scripts.append({
                    'type': 'inline',
                    'content': script_content[:200] + '...' if len(script_content) > 200 else script_content,
                    'line_start': content[:match.start()].count('\n') + 1
                })
        
        # Scripts externos
        src_pattern = r'<script[^>]*src=["\'](.*?)["\'][^>]*>'
        for match in re.finditer(src_pattern, content, re.IGNORECASE):
            scripts.append({
                'type': 'external',
                'src': match.group(1),
                'line_start': content[:match.start()].count('\n') + 1
            })
        
        return scripts
    
    def _extract_styles(self, content: str) -> List[Dict[str, Any]]:
        """Extrae estilos CSS inline y externos."""
        styles = []
        
        # Estilos inline
        style_pattern = r'<style[^>]*>(.*?)</style>'
        for match in re.finditer(style_pattern, content, re.IGNORECASE | re.DOTALL):
            style_content = match.group(1).strip()
            if style_content:
                styles.append({
                    'type': 'inline',
                    'content': style_content[:200] + '...' if len(style_content) > 200 else style_content,
                    'line_start': content[:match.start()].count('\n') + 1
                })
        
        # Estilos externos
        link_pattern = r'<link[^>]*href=["\'](.*?\.css[^"\']*)["\'][^>]*>'
        for match in re.finditer(link_pattern, content, re.IGNORECASE):
            styles.append({
                'type': 'external',
                'href': match.group(1),
                'line_start': content[:match.start()].count('\n') + 1
            })
        
        return styles
    
    def _extract_links(self, content: str) -> List[Dict[str, str]]:
        """Extrae enlaces del documento."""
        links = []
        pattern = r'<a[^>]*href=["\'](.*?)["\'][^>]*>(.*?)</a>'
        
        for match in re.finditer(pattern, content, re.IGNORECASE | re.DOTALL):
            links.append({
                'href': match.group(1),
                'text': re.sub(r'<[^>]+>', '', match.group(2)).strip()
            })
        
        return links
    
    def _extract_images(self, content: str) -> List[Dict[str, str]]:
        """Extrae imágenes del documento."""
        images = []
        pattern = r'<img[^>]*src=["\'](.*?)["\'][^>]*>'
        
        for match in re.finditer(pattern, content, re.IGNORECASE):
            images.append({
                'src': match.group(1),
                'alt': self._extract_attr(match.group(0), 'alt')
            })
        
        return images
    
    def _extract_forms(self, content: str) -> List[Dict[str, Any]]:
        """Extrae formularios."""
        forms = []
        pattern = r'<form[^>]*>(.*?)</form>'
        
        for match in re.finditer(pattern, content, re.IGNORECASE | re.DOTALL):
            form_content = match.group(1)
            forms.append({
                'method': self._extract_attr(match.group(0), 'method'),
                'action': self._extract_attr(match.group(0), 'action'),
                'inputs': len(re.findall(r'<input', form_content, re.IGNORECASE)),
                'line_start': content[:match.start()].count('\n') + 1
            })
        
        return forms
    
    def _extract_meta_tags(self, content: str) -> List[Dict[str, str]]:
        """Extrae meta tags."""
        meta_tags = []
        pattern = r'<meta[^>]*>'
        
        for match in re.finditer(pattern, content, re.IGNORECASE):
            meta = match.group(0)
            meta_tags.append({
                'name': self._extract_attr(meta, 'name'),
                'content': self._extract_attr(meta, 'content'),
                'property': self._extract_attr(meta, 'property')
            })
        
        return meta_tags
    
    def _extract_attr(self, tag: str, attr: str) -> str:
        """Extrae un atributo de una etiqueta HTML."""
        pattern = f'{attr}=["\'](.*?)["\']'
        match = re.search(pattern, tag, re.IGNORECASE)
        return match.group(1) if match else ''