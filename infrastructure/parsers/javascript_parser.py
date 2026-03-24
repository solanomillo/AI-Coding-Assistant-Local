"""
Parser para JavaScript/TypeScript.
Soporta detección de funciones, clases y estructuras básicas.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

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
            extensions=['.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs']
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
        result = {
            'functions': self.extract_functions(content),
            'classes': self.extract_classes(content),
            'imports': self.extract_imports(content),
            'variables': self.extract_variables(content),
            'language': self.language
        }
        
        logger.debug(f"Archivo {file_path.name}: {len(result['functions'])} funciones, "
                    f"{len(result['classes'])} clases, {len(result['imports'])} imports")
        
        return result
    
    def extract_functions(self, content: str) -> List[Dict[str, Any]]:
        """
        Extrae funciones del contenido.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            Lista de funciones encontradas
        """
        functions = []
        
        # Patrón para función declaración: function nombre() {}
        func_declaration = re.compile(
            r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)\s*\{',
            re.MULTILINE
        )
        
        for match in func_declaration.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            functions.append({
                'name': match.group(1),
                'line_start': line_num,
                'line_end': line_num + content[match.start():].count('\n', 0, 500)  # Estimación
            })
        
        # Patrón para función flecha asignada: const nombre = () => {}
        arrow_assignment = re.compile(
            r'(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*\([^)]*\)\s*=>\s*\{',
            re.MULTILINE
        )
        
        for match in arrow_assignment.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            functions.append({
                'name': match.group(1),
                'line_start': line_num,
                'line_end': line_num + content[match.start():].count('\n', 0, 500)
            })
        
        # Patrón para función flecha con un parámetro sin paréntesis: const nombre = param => {}
        arrow_single = re.compile(
            r'(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=>\s*\{',
            re.MULTILINE
        )
        
        for match in arrow_single.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            functions.append({
                'name': match.group(1),
                'line_start': line_num,
                'line_end': line_num + content[match.start():].count('\n', 0, 500)
            })
        
        # Patrón para métodos en objetos: nombre: function() {}
        method_pattern = re.compile(
            r'([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:\s*function\s*\([^)]*\)\s*\{',
            re.MULTILINE
        )
        
        for match in method_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            functions.append({
                'name': match.group(1),
                'line_start': line_num,
                'line_end': line_num + content[match.start():].count('\n', 0, 500)
            })
        
        # Patrón para métodos shorthand: nombre() {}
        shorthand_method = re.compile(
            r'([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)\s*\{',
            re.MULTILINE
        )
        
        # Filtrar para no duplicar
        existing_names = {f['name'] for f in functions}
        for match in shorthand_method.finditer(content):
            name = match.group(1)
            if name not in existing_names:
                line_num = content[:match.start()].count('\n') + 1
                functions.append({
                    'name': name,
                    'line_start': line_num,
                    'line_end': line_num + content[match.start():].count('\n', 0, 500)
                })
                existing_names.add(name)
        
        return functions
    
    def extract_classes(self, content: str) -> List[Dict[str, Any]]:
        """
        Extrae clases del contenido.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            Lista de clases encontradas
        """
        classes = []
        
        # Patrón para clase: class Nombre { ... }
        class_pattern = re.compile(
            r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(?:extends\s+([a-zA-Z_$][a-zA-Z0-9_$]*))?\s*\{',
            re.MULTILINE
        )
        
        for match in class_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            class_name = match.group(1)
            parent_class = match.group(2) if match.group(2) else None
            
            # Extraer métodos de la clase
            methods = []
            class_start = match.end()
            # Buscar el cierre de la clase
            brace_count = 1
            class_end = class_start
            for i in range(class_start, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        class_end = i
                        break
            
            class_body = content[class_start:class_end]
            
            # Buscar métodos dentro de la clase
            method_pattern = re.compile(
                r'(?:async\s+)?([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)\s*\{',
                re.MULTILINE
            )
            
            for m in method_pattern.finditer(class_body):
                method_name = m.group(1)
                if method_name not in ['constructor']:
                    methods.append({'name': method_name})
            
            classes.append({
                'name': class_name,
                'line_start': line_num,
                'parent_class': parent_class,
                'methods': methods[:10]
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
        
        # Import ES6: import ... from 'module'
        es6_import = re.compile(
            r'import\s+(?:(?:\{[^}]*\}|\*\s+as\s+\w+|\w+)\s+from\s+)?[\'"]([^\'"]+)[\'"]',
            re.MULTILINE
        )
        
        for match in es6_import.finditer(content):
            imports.append(f"import from {match.group(1)}")
        
        # Import dinámico: import('module')
        dynamic_import = re.compile(
            r'import\([\'"]([^\'"]+)[\'"]\)',
            re.MULTILINE
        )
        
        for match in dynamic_import.finditer(content):
            imports.append(f"dynamic import({match.group(1)})")
        
        # Require CommonJS: require('module')
        require_pattern = re.compile(
            r'require\([\'"]([^\'"]+)[\'"]\)',
            re.MULTILINE
        )
        
        for match in require_pattern.finditer(content):
            imports.append(f"require({match.group(1)})")
        
        return imports
    
    def extract_variables(self, content: str) -> List[Dict[str, Any]]:
        """
        Extrae variables globales o de exportación.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            Lista de variables
        """
        variables = []
        
        # Exportaciones: export const nombre = ...
        export_pattern = re.compile(
            r'export\s+(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)',
            re.MULTILINE
        )
        
        for match in export_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            variables.append({
                'name': match.group(1),
                'line_start': line_num
            })
        
        return variables