"""
Parser especializado para archivos Python usando AST.

Este módulo proporciona funcionalidades para analizar archivos Python
y extraer información estructurada sobre funciones, clases y metadatos.
"""

import ast
from pathlib import Path
from typing import List, Optional, Tuple, Any
import logging
from domain.models.repository import CodeFile, Function, Class

logger = logging.getLogger(__name__)


class PythonParser:
    """
    Analizador de código Python usando el módulo AST.
    
    Esta clase extrae información estructurada de archivos Python
    incluyendo funciones, clases, docstrings y métricas de complejidad.
    """
    
    def __init__(self):
        """Inicializa el parser de Python."""
        self.supported_extensions = {'.py'}
    
    def can_parse(self, file_path: Path) -> bool:
        """
        Verifica si el archivo puede ser parseado por este parser.
        
        Args:
            file_path: Ruta del archivo a verificar
            
        Returns:
            True si el archivo tiene extensión .py
        """
        return file_path.suffix.lower() in self.supported_extensions
    
    def parse_file(self, file_path: Path, content: str) -> Optional[CodeFile]:
        """
        Parsea un archivo Python y extrae su información.
        
        Args:
            file_path: Ruta del archivo
            content: Contenido del archivo
            
        Returns:
            CodeFile con la información extraída o None si hay error
            
        Raises:
            SyntaxError: Si el archivo tiene errores de sintaxis
        """
        try:
            # Crear árbol AST
            tree = ast.parse(content)
            
            # Crear objeto CodeFile
            code_file = CodeFile(
                path=file_path,
                extension=file_path.suffix,
                line_count=len(content.splitlines()),
                content_hash=code_file.calculate_hash(content) if 'code_file' in locals() else None,
                last_modified=file_path.stat().st_mtime if file_path.exists() else None
            )
            
            # Calcular hash
            code_file.content_hash = code_file.calculate_hash(content)
            
            # Extraer imports
            code_file.imports = self._extract_imports(tree)
            
            # Extraer funciones
            code_file.functions = self._extract_functions(tree)
            
            # Extraer clases
            code_file.classes = self._extract_classes(tree)
            
            logger.debug(f"Archivo parseado: {file_path.name} - "
                        f"{len(code_file.functions)} funciones, "
                        f"{len(code_file.classes)} clases")
            
            return code_file
            
        except SyntaxError as e:
            logger.error(f"Error de sintaxis en {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parseando {file_path}: {e}")
            return None
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """
        Extrae todas las importaciones del árbol AST.
        
        Args:
            tree: Árbol AST
            
        Returns:
            Lista de strings con las importaciones
        """
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
    
    def _extract_functions(self, tree: ast.AST) -> List[Function]:
        """
        Extrae todas las funciones del árbol AST.
        
        Args:
            tree: Árbol AST
            
        Returns:
            Lista de funciones encontradas
        """
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extraer docstring
                docstring = ast.get_docstring(node)
                
                # Extraer argumentos
                arguments = [arg.arg for arg in node.args.args]
                
                # Extraer decoradores
                decorators = [self._get_decorator_name(d) for d in node.decorator_list]
                
                # Calcular complejidad básica
                complexity = self._calculate_complexity(node)
                
                function = Function(
                    name=node.name,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    docstring=docstring,
                    complexity=complexity,
                    decorators=decorators,
                    arguments=arguments
                )
                functions.append(function)
        
        return functions
    
    def _extract_classes(self, tree: ast.AST) -> List[Class]:
        """
        Extrae todas las clases del árbol AST.
        
        Args:
            tree: Árbol AST
            
        Returns:
            Lista de clases encontradas
        """
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
                
                # Extraer decoradores
                decorators = [self._get_decorator_name(d) for d in node.decorator_list]
                
                # Extraer métodos
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_docstring = ast.get_docstring(item)
                        method = Function(
                            name=item.name,
                            line_start=item.lineno,
                            line_end=item.end_lineno or item.lineno,
                            docstring=method_docstring,
                            complexity=self._calculate_complexity(item)
                        )
                        methods.append(method)
                
                class_obj = Class(
                    name=node.name,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    docstring=docstring,
                    parent_class=parent_class,
                    methods=methods,
                    decorators=decorators
                )
                classes.append(class_obj)
        
        return classes
    
    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """
        Extrae el nombre de un decorador.
        
        Args:
            decorator: Nodo AST del decorador
            
        Returns:
            Nombre del decorador como string
        """
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self._get_decorator_name(decorator.value)}.{decorator.attr}"
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        else:
            return str(decorator)
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """
        Calcula una complejidad ciclomática básica.
        
        Args:
            node: Nodo AST
            
        Returns:
            Valor de complejidad (1 + puntos de decisión)
        """
        complexity = 1  # Complejidad base
        
        for child in ast.walk(node):
            # Puntos de decisión que aumentan complejidad
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
        
        return complexity