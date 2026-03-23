"""
Servicio de embeddings usando exclusivamente Gemini.

Este módulo maneja la generación de embeddings para código
utilizando la API de Google Gemini.
"""

import logging
from typing import List, Optional
import google.generativeai as genai
import os
from dotenv import load_dotenv
import hashlib
import re
import time

load_dotenv()
logger = logging.getLogger(__name__)


class GeminiEmbedding:
    """
    Servicio para generar embeddings usando Gemini.
    
    Utiliza el modelo 'models/gemini-embedding-001' de Gemini para
    generar embeddings de texto y código.
    
    NOTA: El modelo embedding-001 retorna vectores de 3072 dimensiones.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el servicio de embeddings de Gemini.
        
        Args:
            api_key: API key de Gemini (opcional, usa .env por defecto)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no encontrada. Configúrala en .env")
        
        # Configurar Gemini
        genai.configure(api_key=self.api_key)
        
        # Modelo de embeddings de Gemini
        self.model = "models/gemini-embedding-001"
        self.dimension = 3072  # Dimensión real del modelo embedding-001
        
        logger.info(f"Gemini Embedding inicializado (dimensión: {self.dimension})")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Genera embedding para un texto usando Gemini.
        
        Args:
            text: Texto a convertir en embedding
            
        Returns:
            Lista de floats con el embedding (dimensión 3072)
        """
        # Validar entrada
        if not text or not isinstance(text, str):
            logger.error(f"Texto inválido para embedding: {type(text)}")
            raise ValueError("Texto inválido para embedding")
        
        # Limitar longitud para evitar problemas
        if len(text) > 2000:
            text = text[:2000]
            logger.debug(f"Texto truncado a 2000 caracteres")
        
        # Verificar que el texto no esté vacío después de limpiar
        text_clean = text.strip()
        if not text_clean:
            logger.warning("Texto vacío para embedding")
            return [0.0] * self.dimension
        
        try:
            logger.debug(f"Generando embedding para texto de {len(text_clean)} caracteres")
            
            # Llamar a la API de Gemini
            result = genai.embed_content(
                model=self.model,
                content=text_clean,
                task_type="retrieval_document"
            )
            
            # Extraer embedding
            embedding = result.get('embedding', [])
            
            # Verificar que se obtuvo un embedding
            if not embedding:
                logger.error("Embedding vacío retornado por la API")
                raise ValueError("Embedding vacío")
            
            # Verificar dimensión
            if len(embedding) != self.dimension:
                logger.error(f"Dimensión incorrecta: {len(embedding)} != {self.dimension}")
                logger.error(f"Embedding recibido: {embedding[:10]}...")
                raise ValueError(f"Dimensión incorrecta: {len(embedding)} != {self.dimension}")
            
            logger.debug(f"Embedding generado correctamente: {len(embedding)} dimensiones")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generando embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para múltiples textos.
        
        Args:
            texts: Lista de textos a convertir
            
        Returns:
            Lista de embeddings
        """
        embeddings = []
        for i, text in enumerate(texts):
            try:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error en texto {i}: {text[:50]}... - {e}")
                embeddings.append([0.0] * self.dimension)
        
        logger.info(f"Generados {len(embeddings)} embeddings en batch")
        return embeddings
    
    def get_dimension(self) -> int:
        """
        Retorna la dimensión real de los embeddings generados.
        
        Returns:
            Dimensión de los embeddings (3072 para embedding-001)
        """
        return self.dimension
    
    @staticmethod
    def chunk_code(code: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Divide código en fragmentos para embedding.
        
        Args:
            code: Código fuente
            chunk_size: Tamaño en caracteres
            overlap: Superposición entre fragmentos
            
        Returns:
            Lista de fragmentos
        """
        chunks = []
        start = 0
        code_length = len(code)
        
        while start < code_length:
            end = min(start + chunk_size, code_length)
            
            # Buscar fin de línea para no cortar en medio
            if end < code_length:
                next_newline = code.find('\n', end)
                if next_newline != -1 and next_newline - end < 100:
                    end = next_newline + 1
            
            chunk = code[start:end]
            if chunk.strip():
                chunks.append(chunk)
            
            # Avanzar con superposición
            start = end - overlap
        
        logger.info(f"Código dividido en {len(chunks)} fragmentos")
        return chunks
    
    @staticmethod
    def create_chunk_id(file_path: str, index: int, content_hash: str) -> str:
        """
        Crea ID único para fragmento.
        
        Args:
            file_path: Ruta del archivo
            index: Índice del fragmento
            content_hash: Hash del contenido
            
        Returns:
            ID único
        """
        unique_str = f"{file_path}:{index}:{content_hash}"
        return hashlib.md5(unique_str.encode()).hexdigest()