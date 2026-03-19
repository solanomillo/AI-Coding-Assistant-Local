"""
Parser para JavaScript/TypeScript usando tree-sitter.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import tree_sitter_javascript as tsjavascript
from tree_sitter import Language, Parser

from infrastructure.parsers.base_parser import BaseParser

logger = logging.getLogger(__name__)


class JavaScriptParser(BaseParser):
    """
    Parser para JavaScript y TypeScript usando tree-sitter.
    Extrae funciones, clases y dependencias.
    """
    
    def __init__(self):
        """Inicializa el parser de JavaScript."""
        super().__init__(
            language="javascript",
            extensions=['.js', '.jsx', '.ts', '.tsx']
        )
        
        # Inicializar tree-sitter para JavaScript
        try:
            self.js_language = Language(tsjavascript.language())
            self.parser = Parser(self.js_language)
            logger.info("Parser JavaScript inicializado con tree-sitter")
        except Exception as e:
            logger.error(f"Error inicializando tree-sitter para JavaScript: {e}")
            self.parser = None
    
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
            'functions': [],
            'classes': [],
            'imports': [],
            'exports': []
        }
        
        if not self.parser:
            logger.warning("Parser no disponible, usando análisis básico")
            return self._basic_parse(content)
        
        try:
            # Parsear con tree-sitter
            tree = self.parser.parse(bytes(content, 'utf-8'))
            
            if tree and tree.root_node:
                # Extraer funciones
                result['functions'] = self._extract_functions(tree.root_node, content)
                
                # Extraer clases
                result['classes'] = self._extract_classes(tree.root_node, content)
                
                # Extraer imports
                result['imports'] = self._extract_imports(tree.root_node, content)
                
                # Extraer exports
                result['exports'] = self._extract_exports(tree.root_node, content)
            
            logger.debug(f"Archivo {file_path.name}: {len(result['functions'])} funciones, "
                        f"{len(result['classes'])} clases")
            
        except Exception as e:
            logger.error(f"Error en parseo tree-sitter: {e}")
            return self._basic_parse(content)
        
        return result
    
    def _extract_functions(self, node, content: str) -> List[Dict[str, Any]]:
        """Extrae funciones del árbol AST."""
        functions = []
        
        def visit(node):
            # Función declaración: function name() {}
            if node.type == 'function_declaration':
                name_node = node.child_by_field_name('name')
                params_node = node.child_by_field_name('parameters')
                
                name = self._get_node_text(name_node, content) if name_node else 'anonymous'
                params = self._get_node_text(params_node, content) if params_node else '()'
                
                functions.append({
                    'name': name,
                    'line_start': node.start_point[0] + 1,
                    'line_end': node.end_point[0] + 1,
                    'parameters': params,
                    'type': 'function'
                })
            
            # Función flecha: const name = () => {}
            elif node.type == 'arrow_function':
                # Buscar nombre en variable declaración
                parent = node.parent
                if parent and parent.type == 'variable_declarator':
                    name_node = parent.child_by_field_name('name')
                    name = self._get_node_text(name_node, content) if name_node else 'anonymous'
                    
                    functions.append({
                        'name': name,
                        'line_start': node.start_point[0] + 1,
                        'line_end': node.end_point[0] + 1,
                        'parameters': '()',
                        'type': 'arrow'
                    })
            
            # Método de clase
            elif node.type == 'method_definition':
                name_node = node.child_by_field_name('name')
                name = self._get_node_text(name_node, content) if name_node else 'method'
                
                functions.append({
                    'name': name,
                    'line_start': node.start_point[0] + 1,
                    'line_end': node.end_point[0] + 1,
                    'parameters': '()',
                    'type': 'method'
                })
            
            # Recorrer hijos
            for child in node.children:
                visit(child)
        
        visit(node)
        return functions
    
    def _extract_classes(self, node, content: str) -> List[Dict[str, Any]]:
        """Extrae clases del árbol AST."""
        classes = []
        
        def visit(node):
            if node.type == 'class_declaration':
                name_node = node.child_by_field_name('name')
                name = self._get_node_text(name_node, content) if name_node else 'anonymous'
                
                # Extraer métodos
                methods = []
                body = node.child_by_field_name('body')
                if body:
                    for child in body.children:
                        if child.type == 'method_definition':
                            method_name = self._get_node_text(
                                child.child_by_field_name('name'), content
                            ) or 'method'
                            methods.append(method_name)
                
                classes.append({
                    'name': name,
                    'line_start': node.start_point[0] + 1,
                    'line_end': node.end_point[0] + 1,
                    'methods': methods[:10]  # Limitar a 10 métodos
                })
            
            for child in node.children:
                visit(child)
        
        visit(node)
        return classes
    
    def _extract_imports(self, node, content: str) -> List[str]:
        """Extrae declaraciones import/require."""
        imports = []
        
        def visit(node):
            # import ... from 'module'
            if node.type == 'import_statement':
                imports.append(self._get_node_text(node, content))
            
            # const x = require('module')
            elif node.type == 'call_expression':
                func = node.child_by_field_name('function')
                if func and self._get_node_text(func, content) == 'require':
                    imports.append(self._get_node_text(node, content))
            
            for child in node.children:
                visit(child)
        
        visit(node)
        return imports
    
    def _extract_exports(self, node, content: str) -> List[str]:
        """Extrae declaraciones export."""
        exports = []
        
        def visit(node):
            if node.type in ['export_statement', 'export_default_declaration']:
                exports.append(self._get_node_text(node, content))
            
            for child in node.children:
                visit(child)
        
        visit(node)
        return exports
    
    def _get_node_text(self, node, content: str) -> str:
        """Obtiene el texto de un nodo."""
        if not node:
            return ''
        start_byte = node.start_byte
        end_byte = node.end_byte
        return content[start_byte:end_byte]
    
    def _basic_parse(self, content: str) -> Dict[str, Any]:
        """
        Parseo básico cuando tree-sitter no está disponible.
        Usa expresiones regulares simples.
        """
        import re
        
        functions = []
        
        # Buscar function nombre() {
        func_pattern = r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\([^)]*\)\s*\{'
        for match in re.finditer(func_pattern, content):
            functions.append({
                'name': match.group(1),
                'line_start': content[:match.start()].count('\n') + 1,
                'type': 'function'
            })
        
        # Buscar const nombre = () => {
        arrow_pattern = r'(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*\([^)]*\)\s*=>\s*\{'
        for match in re.finditer(arrow_pattern, content):
            functions.append({
                'name': match.group(1),
                'line_start': content[:match.start()].count('\n') + 1,
                'type': 'arrow'
            })
        
        # Buscar clases
        classes = []
        class_pattern = r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)'
        for match in re.finditer(class_pattern, content):
            classes.append({
                'name': match.group(1),
                'line_start': content[:match.start()].count('\n') + 1
            })
        
        return {
            'functions': functions,
            'classes': classes,
            'imports': [],
            'exports': []
        }