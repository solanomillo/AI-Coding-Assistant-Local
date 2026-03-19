"""
Modelos de dominio para repositorios y archivos de código.
CORREGIDO: Agregado setter para relative_path
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import hashlib


@dataclass
class Function:
    """Representa una función o método en el código."""
    
    name: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    complexity: int = 1
    decorators: List[str] = field(default_factory=list)
    arguments: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la función a diccionario."""
        return {
            'name': self.name,
            'line_start': self.line_start,
            'line_end': self.line_end,
            'docstring': self.docstring,
            'complexity': self.complexity,
            'decorators': self.decorators,
            'arguments': self.arguments
        }


@dataclass
class Class:
    """Representa una clase en el código."""
    
    name: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    parent_class: Optional[str] = None
    methods: List[Function] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la clase a diccionario."""
        return {
            'name': self.name,
            'line_start': self.line_start,
            'line_end': self.line_end,
            'docstring': self.docstring,
            'parent_class': self.parent_class,
            'methods': [m.to_dict() for m in self.methods],
            'decORAORS': self.decorators
        }


@dataclass
class CodeFile:
    """Representa un archivo de código en el repositorio."""
    
    path: Path
    extension: str
    line_count: int = 0
    functions: List[Function] = field(default_factory=list)
    classes: List[Class] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    content_hash: Optional[str] = None
    last_modified: Optional[datetime] = None
    _relative_path: Optional[str] = None  # Campo privado para almacenar la ruta relativa
    
    @property
    def name(self) -> str:
        """Retorna el nombre del archivo."""
        return self.path.name
    
    @property
    def relative_path(self) -> str:
        """
        Retorna la ruta relativa del archivo.
        Si no se ha establecido explícitamente, usa la ruta del path.
        """
        if self._relative_path:
            return self._relative_path
        return str(self.path)
    
    @relative_path.setter
    def relative_path(self, value: str) -> None:
        """
        Establece la ruta relativa del archivo.
        
        Args:
            value: Ruta relativa a establecer
        """
        self._relative_path = value
    
    def calculate_hash(self, content: str) -> str:
        """
        Calcula el hash SHA-256 del contenido.
        
        Args:
            content: Contenido del archivo
            
        Returns:
            Hash SHA-256 en hexadecimal
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el archivo a diccionario."""
        return {
            'name': self.name,
            'path': self.relative_path,
            'extension': self.extension,
            'line_count': self.line_count,
            'functions': [f.to_dict() for f in self.functions],
            'classes': [c.to_dict() for c in self.classes],
            'imports': self.imports,
            'content_hash': self.content_hash,
            'last_modified': self.last_modified.isoformat() if self.last_modified else None
        }


@dataclass
class Repository:
    """Representa un repositorio de código completo."""
    
    name: str
    path: Path
    language: str = "python"
    files: List[CodeFile] = field(default_factory=list)
    file_count: int = 0
    total_lines: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_analyzed: Optional[datetime] = None
    
    def add_file(self, file: CodeFile) -> None:
        """
        Añade un archivo al repositorio.
        
        Args:
            file: Archivo a añadir
        """
        self.files.append(file)
        self.file_count = len(self.files)
        self.total_lines += file.line_count
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Genera un resumen del repositorio.
        
        Returns:
            Diccionario con estadísticas
        """
        total_functions = sum(len(f.functions) for f in self.files)
        total_classes = sum(len(c.classes) for c in self.files)
        
        return {
            'name': self.name,
            'path': str(self.path),
            'language': self.language,
            'file_count': self.file_count,
            'total_lines': self.total_lines,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'created_at': self.created_at.isoformat(),
            'last_analyzed': self.last_analyzed.isoformat() if self.last_analyzed else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el repositorio a diccionario."""
        return {
            'name': self.name,
            'path': str(self.path),
            'language': self.language,
            'files': [f.to_dict() for f in self.files],
            'file_count': self.file_count,
            'total_lines': self.total_lines,
            'created_at': self.created_at.isoformat(),
            'last_analyzed': self.last_analyzed.isoformat() if self.last_analyzed else None
        }