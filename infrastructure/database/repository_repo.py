"""
Repositorio de base de datos para operaciones con repositorios.

Este módulo maneja todas las operaciones CRUD con MySQL para
repositorios, archivos, funciones y clases.
"""

import pymysql
import pymysql.cursors
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

from domain.models.repository import Repository, CodeFile, Function, Class

load_dotenv()
logger = logging.getLogger(__name__)


class MySQLRepository:
    """
    Repositorio MySQL para persistencia de datos de repositorios.
    
    Esta clase maneja todas las operaciones de base de datos relacionadas
    con repositorios, archivos y sus componentes.
    """
    
    def __init__(self):
        """Inicializa la conexión a MySQL con configuración de entorno."""
        self.config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'ai_coding_assistant'),
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor
        }
        self._test_connection()
    
    def _get_connection(self):
        """
        Obtiene una conexión a la base de datos.
        
        Returns:
            Conexión MySQL
        """
        return pymysql.connect(**self.config)
    
    def _test_connection(self) -> None:
        """Prueba la conexión a la base de datos."""
        try:
            with self._get_connection() as conn:
                conn.ping()
            logger.info("Conexión a MySQL establecida correctamente")
        except pymysql.Error as e:
            logger.error(f"Error conectando a MySQL: {e}")
            raise
    
    def save_repository(self, repository: Repository) -> int:
        """
        Guarda un repositorio en la base de datos.
        
        Args:
            repository: Repositorio a guardar
            
        Returns:
            ID del repositorio insertado
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Insertar repositorio
                    sql = """
                        INSERT INTO repositories 
                        (name, path, language, file_count, total_lines, last_analyzed)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(sql, (
                        repository.name,
                        str(repository.path),
                        repository.language,
                        repository.file_count,
                        repository.total_lines,
                        datetime.now()
                    ))
                    
                    repo_id = cursor.lastrowid
                    
                    # Guardar archivos
                    for file in repository.files:
                        self._save_file(conn, repo_id, file)
                    
                    conn.commit()
                    logger.info(f"Repositorio {repository.name} guardado con ID: {repo_id}")
                    return repo_id
                    
        except pymysql.Error as e:
            logger.error(f"Error guardando repositorio: {e}")
            raise
    
    def _save_file(self, conn, repository_id: int, file: CodeFile) -> int:
        """
        Guarda un archivo en la base de datos.
        
        Args:
            conn: Conexión MySQL
            repository_id: ID del repositorio
            file: Archivo a guardar
            
        Returns:
            ID del archivo insertado
        """
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO files 
                (repository_id, file_path, file_name, extension, 
                 line_count, function_count, class_count, content_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                repository_id,
                file.relative_path,
                file.name,
                file.extension,
                file.line_count,
                len(file.functions),
                len(file.classes),
                file.content_hash
            ))
            
            file_id = cursor.lastrowid
            
            # Guardar funciones
            for func in file.functions:
                self._save_function(conn, file_id, func)
            
            # Guardar clases
            for cls in file.classes:
                self._save_class(conn, file_id, cls)
            
            return file_id
    
    def _save_function(self, conn, file_id: int, function: Function) -> int:
        """
        Guarda una función en la base de datos.
        
        Args:
            conn: Conexión MySQL
            file_id: ID del archivo
            function: Función a guardar
            
        Returns:
            ID de la función insertada
        """
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO functions 
                (file_id, name, line_start, line_end, docstring, complexity)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                file_id,
                function.name,
                function.line_start,
                function.line_end,
                function.docstring,
                function.complexity
            ))
            return cursor.lastrowid
    
    def _save_class(self, conn, file_id: int, class_obj: Class) -> int:
        """
        Guarda una clase en la base de datos.
        
        Args:
            conn: Conexión MySQL
            file_id: ID del archivo
            class_obj: Clase a guardar
            
        Returns:
            ID de la clase insertada
        """
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO classes 
                (file_id, name, line_start, line_end, docstring, parent_class)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                file_id,
                class_obj.name,
                class_obj.line_start,
                class_obj.line_end,
                class_obj.docstring,
                class_obj.parent_class
            ))
            
            class_id = cursor.lastrowid
            
            # Guardar métodos de la clase
            for method in class_obj.methods:
                self._save_function(conn, file_id, method)
            
            return class_id
    
    def get_repository(self, repo_id: int) -> Optional[Dict[str, Any]]:
        """
        Obtiene un repositorio por su ID.
        
        Args:
            repo_id: ID del repositorio
            
        Returns:
            Diccionario con datos del repositorio o None
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = "SELECT * FROM repositories WHERE id = %s"
                    cursor.execute(sql, (repo_id,))
                    return cursor.fetchone()
        except pymysql.Error as e:
            logger.error(f"Error obteniendo repositorio {repo_id}: {e}")
            return None
    
    def get_repository_by_path(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene un repositorio por su ruta.
        
        Args:
            path: Ruta del repositorio
            
        Returns:
            Diccionario con datos del repositorio o None
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = "SELECT * FROM repositories WHERE path = %s"
                    cursor.execute(sql, (path,))
                    return cursor.fetchone()
        except pymysql.Error as e:
            logger.error(f"Error obteniendo repositorio por ruta {path}: {e}")
            return None
    
    def list_repositories(self) -> List[Dict[str, Any]]:
        """
        Lista todos los repositorios.
        
        Returns:
            Lista de repositorios
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = """
                        SELECT id, name, path, language, file_count, 
                               total_lines, created_at, last_analyzed
                        FROM repositories 
                        ORDER BY created_at DESC
                    """
                    cursor.execute(sql)
                    return cursor.fetchall()
        except pymysql.Error as e:
            logger.error(f"Error listando repositorios: {e}")
            return []
    
    def delete_repository(self, repo_id: int) -> bool:
        """
        Elimina un repositorio (cascada elimina archivos, funciones, clases).
        
        Args:
            repo_id: ID del repositorio
            
        Returns:
            True si se eliminó correctamente
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = "DELETE FROM repositories WHERE id = %s"
                    cursor.execute(sql, (repo_id,))
                    conn.commit()
                    return cursor.rowcount > 0
        except pymysql.Error as e:
            logger.error(f"Error eliminando repositorio {repo_id}: {e}")
            return False
    
    def update_analysis_time(self, repo_id: int) -> bool:
        """
        Actualiza el timestamp del último análisis.
        
        Args:
            repo_id: ID del repositorio
            
        Returns:
            True si se actualizó correctamente
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = "UPDATE repositories SET last_analyzed = %s WHERE id = %s"
                    cursor.execute(sql, (datetime.now(), repo_id))
                    conn.commit()
                    return cursor.rowcount > 0
        except pymysql.Error as e:
            logger.error(f"Error actualizando análisis de {repo_id}: {e}")
            return False