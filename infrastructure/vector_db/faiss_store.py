"""
Almacén vectorial usando FAISS.

Este módulo implementa una base de datos vectorial liviana
usando FAISS para búsqueda de similitud.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
import json
import numpy as np
import faiss
import traceback

logger = logging.getLogger(__name__)


class FAISSStore:
    """
    Almacén vectorial usando FAISS.
    
    Características:
    - Índice plano (exacto) para búsqueda precisa
    - Persistencia en disco
    - Metadatos asociados a vectores
    """
    
    def __init__(self, dimension: int = 3072, index_path: Optional[Path] = None):
        """
        Inicializa el almacén FAISS.
        
        Args:
            dimension: Dimensión de los vectores (3072 para Gemini)
            index_path: Ruta para persistencia
        """
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else Path("data/vector_store/faiss.index")
        self.metadata_path = self.index_path.with_suffix('.pkl')
        
        # Crear directorio si no existe
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Inicializar o cargar índice
        self.index = self._load_or_create_index()
        
        # Metadatos (id -> metadata)
        self.metadata = self._load_metadata()
        
        # Mapeo inverso (id -> posición en índice)
        self.id_to_position = {}
        self.position_to_id = []
        
        # Reconstruir mapeos si hay datos
        self._rebuild_mappings()
        
        logger.info(f"FAISSStore inicializado: {len(self.metadata)} vectores, dimensión {dimension}")
    
    def _load_or_create_index(self) -> faiss.Index:
        """Carga índice existente o crea uno nuevo."""
        try:
            if self.index_path.exists():
                index = faiss.read_index(str(self.index_path))
                logger.info(f"Índice cargado desde {self.index_path}")
                return index
        except Exception as e:
            logger.warning(f"Error cargando índice: {e}")
        
        # Crear índice plano (búsqueda exacta)
        index = faiss.IndexFlatIP(self.dimension)
        logger.info(f"Índice nuevo creado (dimensión: {self.dimension})")
        return index
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Carga metadatos desde archivo."""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                logger.info(f"Metadatos cargados: {len(metadata)} entradas")
                return metadata
        except Exception as e:
            logger.warning(f"Error cargando metadatos: {e}")
        
        return {}
    
    def _save_metadata(self) -> None:
        """Guarda metadatos en disco."""
        try:
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.debug(f"Metadatos guardados en {self.metadata_path}")
        except Exception as e:
            logger.error(f"Error guardando metadatos: {e}")
    
    def _rebuild_mappings(self) -> None:
        """Reconstruye mapeos internos desde metadatos."""
        self.id_to_position = {}
        self.position_to_id = []
        
        for vec_id, data in self.metadata.items():
            position = data.get('position')
            if position is not None:
                self.id_to_position[vec_id] = position
                while len(self.position_to_id) <= position:
                    self.position_to_id.append(None)
                self.position_to_id[position] = vec_id
        
        logger.debug(f"Mapeos reconstruidos: {len(self.id_to_position)} entradas")
    
    def add_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Añade vectores al índice.
        """
        if not vectors:
            logger.warning("No hay vectores para añadir")
            return
        
        if len(vectors) != len(ids) or len(vectors) != len(metadatas):
            raise ValueError(f"Longitudes inconsistentes: vectors={len(vectors)}, ids={len(ids)}, metadatas={len(metadatas)}")
        
        try:
            # Verificar cada vector tiene la dimensión correcta
            for i, vec in enumerate(vectors):
                if len(vec) != self.dimension:
                    raise ValueError(f"Vector {i} tiene dimensión {len(vec)} pero se esperaba {self.dimension}")
            
            # Convertir a numpy array
            vectors_np = np.array(vectors, dtype=np.float32)
            logger.debug(f"Array numpy creado: forma={vectors_np.shape}, dtype={vectors_np.dtype}")
            
            # Normalizar para similitud coseno
            faiss.normalize_L2(vectors_np)
            logger.debug("Vectores normalizados")
            
            # Obtener posición actual
            start_pos = self.index.ntotal
            logger.debug(f"Posición inicial: {start_pos}")
            
            # Añadir vectores
            self.index.add(vectors_np)
            logger.debug("Vectores añadidos al índice")
            
            # Guardar metadatos y mapeos
            for i, (vec_id, metadata) in enumerate(zip(ids, metadatas)):
                position = start_pos + i
                self.metadata[vec_id] = {
                    'position': position,
                    'data': metadata
                }
                self.id_to_position[vec_id] = position
                
                while len(self.position_to_id) <= position:
                    self.position_to_id.append(None)
                self.position_to_id[position] = vec_id
            
            # Guardar metadatos
            self._save_metadata()
            
            # Guardar índice
            faiss.write_index(self.index, str(self.index_path))
            
            logger.info(f"Añadidos {len(vectors)} vectores. Total en índice: {self.index.ntotal}")
            
        except Exception as e:
            logger.error(f"Error añadiendo vectores: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca vectores similares.
        """
        if self.index.ntotal == 0:
            logger.warning("Índice vacío")
            return []
        
        try:
            # Validar dimensión del query
            if len(query_vector) != self.dimension:
                logger.error(f"Query vector dimensión {len(query_vector)} != {self.dimension}")
                return []
            
            # Preparar query
            query_np = np.array([query_vector], dtype=np.float32)
            faiss.normalize_L2(query_np)
            
            # Buscar
            k_actual = min(k, self.index.ntotal)
            scores, indices = self.index.search(query_np, k_actual)
            
            # Formatear resultados
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                
                # Buscar ID por posición
                vec_id = None
                if idx < len(self.position_to_id):
                    vec_id = self.position_to_id[idx]
                
                if vec_id and vec_id in self.metadata:
                    results.append({
                        'id': vec_id,
                        'score': float(score),
                        'metadata': self.metadata[vec_id]['data']
                    })
            
            logger.debug(f"Búsqueda: {len(results)} resultados de {k_actual} solicitados")
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del índice."""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_path': str(self.index_path),
            'metadata_count': len(self.metadata),
            'index_type': type(self.index).__name__
        }
    
    def clear(self) -> None:
        """Limpia todo el índice."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = {}
        self.id_to_position = {}
        self.position_to_id = []
        
        faiss.write_index(self.index, str(self.index_path))
        self._save_metadata()
        
        logger.info("Índice limpiado")