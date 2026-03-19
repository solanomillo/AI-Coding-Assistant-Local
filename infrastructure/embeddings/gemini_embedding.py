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

load_dotenv()
logger = logging.getLogger(__name__)


class GeminiEmbedding:
    """
    Servicio para generar embeddings usando Gemini.
    
    Utiliza el modelo 'models/gemini-embedding-001' de Gemini para
    generar embeddings de texto y código.
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
        self.dimension = 768  # Dimensión fija para embedding-001
        
        logger.info(f"✅ Gemini Embedding inicializado (dimensión: {self.dimension})")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Genera embedding para un texto usando Gemini.
        
        Args:
            text: Texto a convertir en embedding
            
        Returns:
            Lista de floats con el embedding
        """
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            
            embedding = result['embedding']
            logger.debug(f"Embedding generado: {len(embedding)} dimensiones")
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
        for text in texts:
            try:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error en texto: {text[:50]}... - {e}")
                embeddings.append([0.0] * self.dimension)  # Placeholder en error
        
        logger.info(f"Generados {len(embeddings)} embeddings en batch")
        return embeddings
    
    def get_dimension(self) -> int:
        """Retorna la dimensión de los embeddings."""
        return self.dimension
    
    @staticmethod
    def chunk_code(code: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
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
