"""
Repositorio MySQL para persistencia de metadatos.
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
    Repositorio MySQL para metadatos de repositorios.
    
    Almacena:
    - Información de repositorios
    - Metadatos de archivos
    - Funciones y clases extraídas
    """
    
    def __init__(self):
        """Inicializa la conexión a MySQL."""
        self.config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'ai_coding_assistant'),
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor
        }
        self._ensure_connection()
        self._ensure_tables()
        
        logger.info("MySQLRepository inicializado")
        logger.info(f"  Host: {self.config['host']}")
        logger.info(f"  Database: {self.config['database']}")
    
    def _get_connection(self):
        """Obtiene conexión a la base de datos."""
        return pymysql.connect(**self.config)
    
    def _ensure_connection(self) -> None:
        """Asegura que la conexión a MySQL funciona."""
        try:
            with self._get_connection() as conn:
                conn.ping()
            logger.info("Conexión a MySQL establecida")
        except pymysql.Error as e:
            logger.error(f"Error conectando a MySQL: {e}")
            raise
    
    def _ensure_tables(self) -> None:
        """Asegura que las tablas necesarias existen."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SHOW TABLES")
                    tables = [t['Tables_in_' + self.config['database']] for t in cursor.fetchall()]
                    
                    required_tables = ['repositories', 'files', 'functions', 'classes']
                    missing = [t for t in required_tables if t not in tables]
                    
                    if missing:
                        logger.warning(f"Tablas faltantes: {missing}")
                        logger.info("Ejecuta scripts/init_database.sql para crear las tablas")
                    
        except pymysql.Error as e:
            logger.error(f"Error verificando tablas: {e}")
    
    def save_repository(self, repository: Repository) -> int:
        """
        Guarda un repositorio y retorna su ID.
        
        Args:
            repository: Repositorio a guardar
            
        Returns:
            ID del repositorio
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
                    logger.info(f"Repositorio guardado con ID: {repo_id}")
                    return repo_id
                    
        except pymysql.Error as e:
            logger.error(f"Error guardando repositorio: {e}")
            raise
    
    def _save_file(self, conn, repository_id: int, file: CodeFile) -> int:
        """Guarda un archivo y retorna su ID."""
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
        """Guarda una función."""
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
        """Guarda una clase."""
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
            return cursor.lastrowid
    
    def get_repository(self, repo_id: int) -> Optional[Dict[str, Any]]:
        """Obtiene un repositorio por ID."""
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
        """Obtiene un repositorio por ruta."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = "SELECT * FROM repositories WHERE path = %s"
                    cursor.execute(sql, (path,))
                    return cursor.fetchone()
        except pymysql.Error as e:
            logger.error(f"Error obteniendo repositorio por ruta: {e}")
            return None
    
    def list_repositories(self) -> List[Dict[str, Any]]:
        """Lista todos los repositorios."""
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
        """Elimina un repositorio (cascada)."""
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
    
    def get_files(self, repo_id: int) -> List[Dict[str, Any]]:
        """Obtiene todos los archivos de un repositorio."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = """
                        SELECT id, file_path, file_name, extension, 
                               line_count, function_count, class_count, content_hash
                        FROM files 
                        WHERE repository_id = %s
                        ORDER BY file_path
                    """
                    cursor.execute(sql, (repo_id,))
                    return cursor.fetchall()
        except pymysql.Error as e:
            logger.error(f"Error obteniendo archivos: {e}")
            return []
    
    def get_functions(self, file_id: int) -> List[Dict[str, Any]]:
        """Obtiene todas las funciones de un archivo."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = """
                        SELECT id, name, line_start, line_end, docstring, complexity
                        FROM functions 
                        WHERE file_id = %s
                        ORDER BY line_start
                    """
                    cursor.execute(sql, (file_id,))
                    return cursor.fetchall()
        except pymysql.Error as e:
            logger.error(f"Error obteniendo funciones: {e}")
            return []
    
    def get_classes(self, file_id: int) -> List[Dict[str, Any]]:
        """Obtiene todas las clases de un archivo."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = """
                        SELECT id, name, line_start, line_end, docstring, parent_class
                        FROM classes 
                        WHERE file_id = %s
                        ORDER BY line_start
                    """
                    cursor.execute(sql, (file_id,))
                    return cursor.fetchall()
        except pymysql.Error as e:
            logger.error(f"Error obteniendo clases: {e}")
            return []