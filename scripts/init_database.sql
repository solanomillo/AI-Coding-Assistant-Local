-- scripts/init_database.sql
-- Script de inicialización de base de datos para AI Coding Assistant

-- Crear base de datos si no existe
CREATE DATABASE IF NOT EXISTS ai_coding_assistant
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE ai_coding_assistant;

-- Tabla de repositorios
CREATE TABLE IF NOT EXISTS repositories (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    path VARCHAR(500) NOT NULL UNIQUE,
    language VARCHAR(50),
    file_count INT DEFAULT 0,
    total_lines INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_analyzed TIMESTAMP NULL,
    INDEX idx_name (name),
    INDEX idx_language (language)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Tabla de archivos del repositorio
CREATE TABLE IF NOT EXISTS files (
    id INT AUTO_INCREMENT PRIMARY KEY,
    repository_id INT NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    extension VARCHAR(20),
    line_count INT DEFAULT 0,
    function_count INT DEFAULT 0,
    class_count INT DEFAULT 0,
    last_modified TIMESTAMP,
    content_hash VARCHAR(64),
    FOREIGN KEY (repository_id) REFERENCES repositories(id) ON DELETE CASCADE,
    INDEX idx_repository (repository_id),
    INDEX idx_extension (extension)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Tabla de funciones/ métodos
CREATE TABLE IF NOT EXISTS functions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_id INT NOT NULL,
    name VARCHAR(255) NOT NULL,
    line_start INT,
    line_end INT,
    docstring TEXT,
    complexity INT DEFAULT 1,
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
    INDEX idx_file (file_id),
    INDEX idx_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Tabla de clases
CREATE TABLE IF NOT EXISTS classes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_id INT NOT NULL,
    name VARCHAR(255) NOT NULL,
    line_start INT,
    line_end INT,
    docstring TEXT,
    parent_class VARCHAR(255),
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
    INDEX idx_file (file_id),
    INDEX idx_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Tabla de análisis/consultas
CREATE TABLE IF NOT EXISTS queries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    repository_id INT NOT NULL,
    query_text TEXT NOT NULL,
    response_text TEXT,
    query_type ENUM('explain', 'review', 'docs', 'general') DEFAULT 'general',
    tokens_used INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (repository_id) REFERENCES repositories(id) ON DELETE CASCADE,
    INDEX idx_repository (repository_id),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Tabla de embeddings (referencias a vectores)
CREATE TABLE IF NOT EXISTS embeddings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_id INT NOT NULL,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    vector_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
    INDEX idx_file_chunk (file_id, chunk_index),
    UNIQUE KEY unique_file_chunk (file_id, chunk_index)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Crear usuario para la aplicación (opcional)
-- CREATE USER IF NOT EXISTS 'ai_assistant'@'localhost' IDENTIFIED BY 'your_secure_password';
-- GRANT ALL PRIVILEGES ON ai_coding_assistant.* TO 'ai_assistant'@'localhost';
-- FLUSH PRIVILEGES;

-- Verificar creación
SHOW TABLES;