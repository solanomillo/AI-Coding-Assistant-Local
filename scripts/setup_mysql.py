#!/usr/bin/env python
"""
Script de configuración de MySQL para AI Coding Assistant.
Este script verifica la conexión a MySQL y crea las tablas necesarias.
"""

import os
import sys
import pymysql
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

def setup_database():
    """
    Configura la base de datos MySQL para el proyecto.
    
    Returns:
        bool: True si la configuración fue exitosa, False en caso contrario.
    """
    print("🔧 Configurando MySQL para AI Coding Assistant...")
    
    # Configuración desde .env
    config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'ai_coding_assistant'),
        'charset': 'utf8mb4'
    }
    
    try:
        # Conectar sin base de datos específica
        connection = pymysql.connect(
            host=config['host'],
            user=config['user'],
            password=config['password'],
            charset=config['charset']
        )
        
        with connection.cursor() as cursor:
            # Crear base de datos si no existe
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config['database']}")
            print(f"✅ Base de datos '{config['database']}' verificada/creada")
            
            # Seleccionar base de datos
            cursor.execute(f"USE {config['database']}")
            
            # Leer y ejecutar script SQL
            sql_path = Path(__file__).parent / 'init_database.sql'
            with open(sql_path, 'r', encoding='utf-8') as f:
                sql_commands = f.read().split(';')
                
                for command in sql_commands:
                    if command.strip():
                        cursor.execute(command)
                
            print("✅ Tablas creadas/verificadas correctamente")
            
            # Mostrar tablas
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            print("\n📊 Tablas disponibles:")
            for table in tables:
                print(f"   - {table[0]}")
        
        connection.commit()
        connection.close()
        
        print("\n✅ Configuración de MySQL completada exitosamente")
        return True
        
    except pymysql.Error as e:
        print(f"❌ Error de MySQL: {e}")
        print("\n📝 Pasos para solucionar:")
        print("1. Verifica que MySQL esté instalado y corriendo")
        print("2. Verifica las credenciales en .env")
        print("3. Ejecuta: mysql -u root -p")
        return False

def test_connection():
    """
    Prueba la conexión a la base de datos.
    """
    try:
        connection = pymysql.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', ''),
            database=os.getenv('DB_NAME', 'ai_coding_assistant'),
            charset='utf8mb4'
        )
        
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM repositories")
            count = cursor.fetchone()[0]
            print(f"✅ Conexión exitosa. Repositorios en DB: {count}")
        
        connection.close()
        return True
    except pymysql.Error as e:
        print(f"❌ Error de conexión: {e}")
        return False

if __name__ == "__main__":
    if setup_database():
        print("\n🔄 Probando conexión...")
        test_connection()
    else:
        sys.exit(1)