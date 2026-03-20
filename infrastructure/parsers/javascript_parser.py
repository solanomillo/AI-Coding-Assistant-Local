"""
Parser para JavaScript/TypeScript usando tree-sitter.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from infrastructure.parsers.base_parser import BaseParser

logger = logging.getLogger(__name__)


class JavaScriptParser(BaseParser):
    """
    Parser para JavaScript y TypeScript.
    """
    
    def __init__(self):
        """Inicializa el parser de JavaScript."""
        super().__init__(
            language="javascript",
            extensions=['.js', '.jsx', '.ts', '.tsx']
        )
        logger.info("Parser JavaScript inicializado")
    
    def can_parse(self, file_path: Path) -> bool:
        """
        Verifica si el archivo puede ser parseado.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            True si es archivo JavaScript/TypeScript
        """
        return file_path.suffix.lower() in self.extensions
    
    def parse_file(self, file_path: Path, content: str) -> Dict[str, Any]:
        """
        Parsea un archivo JavaScript/TypeScript.
        
        Args:
            file_path: Ruta del archivo
            content: Contenido del archivo
            
        Returns:
            Diccionario con funciones, clases e imports
        """
        return {
            'functions': self.extract_functions(content),
            'classes': self.extract_classes(content),
            'imports': self.extract_imports(content),
            'language': self.language
        }
    
    def extract_functions(self, content: str) -> List[Dict[str, Any]]:
        """
        Extrae funciones del contenido.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            Lista de funciones
        """
        functions = []
        
        # Función declaración: function name() {}
        func_pattern = r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)\s*\{'
        for match in re.finditer(func_pattern, content):
            functions.append({
                'name': match.group(1),
                'line_start': content[:match.start()].count('\n') + 1,
                'type': 'function'
            })
        
        # Función flecha: const name = () => {}
        arrow_pattern = r'(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*\([^)]*\)\s*=>\s*\{'
        for match in re.finditer(arrow_pattern, content):
            functions.append({
                'name': match.group(1),
                'line_start': content[:match.start()].count('\n') + 1,
                'type': 'arrow'
            })
        
        return functions
    
    def extract_classes(self, content: str) -> List[Dict[str, Any]]:
        """
        Extrae clases del contenido.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            Lista de clases
        """
        classes = []
        
        class_pattern = r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)'
        for match in re.finditer(class_pattern, content):
            classes.append({
                'name': match.group(1),
                'line_start': content[:match.start()].count('\n') + 1
            })
        
        return classes
    
    def extract_imports(self, content: str) -> List[str]:
        """
        Extrae importaciones del contenido.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            Lista de imports
        """
        imports = []
        
        # import ... from 'module'
        import_pattern = r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]'
        for match in re.finditer(import_pattern, content):
            imports.append(f"import from {match.group(1)}")
        
        # require('module')
        require_pattern = r'require\([\'"]([^\'"]+)[\'"]\)'
        for match in re.finditer(require_pattern, content):
            imports.append(f"require({match.group(1)})")
        
        return imports