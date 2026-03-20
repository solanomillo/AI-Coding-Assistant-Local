"""
Parser especializado para archivos Python usando AST.
"""

import ast
from pathlib import Path
from typing import List, Optional, Tuple, Any
import logging

from infrastructure.parsers.base_parser import BaseParser

logger = logging.getLogger(__name__)


class PythonParser(BaseParser):
    """
    Analizador de código Python usando el módulo AST.
    """
    
    def __init__(self):
        """Inicializa el parser de Python."""
        super().__init__(
            language="python",
            extensions=['.py']
        )
        logger.info("Parser Python inicializado")
    
    def can_parse(self, file_path: Path) -> bool:
        """
        Verifica si el archivo puede ser parseado.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            True si es archivo Python
        """
        return file_path.suffix.lower() in self.extensions
    
    def parse_file(self, file_path: Path, content: str) -> Optional[dict]:
        """
        Parsea un archivo Python y extrae su información.
        
        Args:
            file_path: Ruta del archivo
            content: Contenido del archivo
            
        Returns:
            Diccionario con funciones, clases e imports
        """
        try:
            # Crear árbol AST
            tree = ast.parse(content)
            
            # Extraer información
            functions = self._extract_functions(tree, content)
            classes = self._extract_classes(tree, content)
            imports = self._extract_imports(tree)
            
            logger.debug(f"Archivo parseado: {file_path.name} - "
                        f"{len(functions)} funciones, {len(classes)} clases")
            
            return {
                'functions': functions,
                'classes': classes,
                'imports': imports,
                'language': self.language
            }
            
        except SyntaxError as e:
            logger.error(f"Error de sintaxis en {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parseando {file_path}: {e}")
            return None
    
    def extract_functions(self, content: str) -> List[dict]:
        """
        Extrae funciones del contenido.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            Lista de funciones
        """
        try:
            tree = ast.parse(content)
            return self._extract_functions(tree, content)
        except Exception as e:
            logger.error(f"Error extrayendo funciones: {e}")
            return []
    
    def extract_classes(self, content: str) -> List[dict]:
        """
        Extrae clases del contenido.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            Lista de clases
        """
        try:
            tree = ast.parse(content)
            return self._extract_classes(tree, content)
        except Exception as e:
            logger.error(f"Error extrayendo clases: {e}")
            return []
    
    def extract_imports(self, content: str) -> List[str]:
        """
        Extrae importaciones del contenido.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            Lista de imports
        """
        try:
            tree = ast.parse(content)
            return self._extract_imports(tree)
        except Exception as e:
            logger.error(f"Error extrayendo imports: {e}")
            return []
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extrae importaciones del árbol AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"from {module} import {alias.name}")
        
        return imports
    
    def _extract_functions(self, tree: ast.AST, content: str) -> List[dict]:
        """Extrae funciones del árbol AST."""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extraer docstring
                docstring = ast.get_docstring(node)
                
                # Extraer argumentos
                arguments = [arg.arg for arg in node.args.args]
                
                # Calcular complejidad básica
                complexity = self._calculate_complexity(node)
                
                function = {
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno or node.lineno,
                    'docstring': docstring,
                    'complexity': complexity,
                    'arguments': arguments
                }
                functions.append(function)
        
        return functions
    
    def _extract_classes(self, tree: ast.AST, content: str) -> List[dict]:
        """Extrae clases del árbol AST."""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Extraer docstring
                docstring = ast.get_docstring(node)
                
                # Extraer clase padre
                parent_class = None
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        parent_class = base.id
                        break
                
                # Extraer métodos
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_docstring = ast.get_docstring(item)
                        method = {
                            'name': item.name,
                            'line_start': item.lineno,
                            'line_end': item.end_lineno or item.lineno,
                            'docstring': method_docstring,
                            'complexity': self._calculate_complexity(item)
                        }
                        methods.append(method)
                
                class_obj = {
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno or node.lineno,
                    'docstring': docstring,
                    'parent_class': parent_class,
                    'methods': methods
                }
                classes.append(class_obj)
        
        return classes
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """
        Calcula complejidad ciclomática básica.
        
        Args:
            node: Nodo AST
            
        Returns:
            Valor de complejidad
        """
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
        
        return complexity