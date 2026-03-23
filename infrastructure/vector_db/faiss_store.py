"""
Almacén vectorial usando FAISS optimizado.
Solo guarda vectores y IDs, el texto completo se almacena en caché.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle
import numpy as np
import faiss
import traceback

logger = logging.getLogger(__name__)


class FAISSStore:
    """
    Almacén vectorial optimizado.
    Guarda solo vectores e IDs, no texto.
    """
    
    def __init__(self, dimension: int = 3072, index_path: Optional[Path] = None):
        """
        Inicializa el almacén FAISS.
        
        Args:
            dimension: Dimensión de los vectores (3072 para Gemini)
            index_path: Ruta para persistencia
        """
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else Path("data/vectors/faiss.index")
        self.metadata_path = self.index_path.with_suffix('.pkl')
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index = self._load_or_create_index()
        self.metadata = self._load_metadata()  # Solo IDs y posiciones
        self._rebuild_mappings()
        
        logger.info(f"FAISSStore inicializado: {len(self.metadata)} vectores, dim {dimension}")
    
    def _load_or_create_index(self) -> faiss.Index:
        """Carga índice existente o crea uno nuevo."""
        try:
            if self.index_path.exists():
                index = faiss.read_index(str(self.index_path))
                logger.info(f"Índice cargado desde {self.index_path}")
                return index
        except Exception as e:
            logger.warning(f"Error cargando índice: {e}")
        
        index = faiss.IndexFlatIP(self.dimension)
        logger.info(f"Índice nuevo creado (dimensión: {self.dimension})")
        return index
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Carga metadatos (solo IDs y posiciones)."""
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
        """Guarda metadatos (solo IDs y posiciones)."""
        try:
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Error guardando metadatos: {e}")
    
    def _rebuild_mappings(self) -> None:
        """Reconstruye mapeos posición-ID."""
        self.id_to_position = {}
        self.position_to_id = []
        
        for vec_id, data in self.metadata.items():
            position = data.get('position')
            if position is not None:
                self.id_to_position[vec_id] = position
                while len(self.position_to_id) <= position:
                    self.position_to_id.append(None)
                self.position_to_id[position] = vec_id
    
    def add_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Añade vectores al índice.
        
        Args:
            vectors: Lista de vectores
            ids: Lista de IDs únicos
            metadatas: Metadatos adicionales (opcional, solo para referencia)
        """
        if not vectors:
            return
        
        if len(vectors) != len(ids):
            raise ValueError(f"Longitudes inconsistentes: vectors={len(vectors)}, ids={len(ids)}")
        
        try:
            # Validar vectores
            for vec in vectors:
                if len(vec) != self.dimension:
                    raise ValueError(f"Vector dimensión {len(vec)} != {self.dimension}")
            
            vectors_np = np.array(vectors, dtype=np.float32)
            faiss.normalize_L2(vectors_np)
            
            start_pos = self.index.ntotal
            self.index.add(vectors_np)
            
            # Guardar solo ID y posición
            for i, vec_id in enumerate(ids):
                position = start_pos + i
                self.metadata[vec_id] = {'position': position}
                self.id_to_position[vec_id] = position
                
                while len(self.position_to_id) <= position:
                    self.position_to_id.append(None)
                self.position_to_id[position] = vec_id
            
            self._save_metadata()
            faiss.write_index(self.index, str(self.index_path))
            
            logger.info(f"Añadidos {len(vectors)} vectores. Total: {self.index.ntotal}")
            
        except Exception as e:
            logger.error(f"Error añadiendo vectores: {e}")
            raise
    
    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Busca vectores similares y retorna solo IDs.
        
        Returns:
            Lista de diccionarios con 'id', 'score', y opcionalmente 'position'
        """
        if self.index.ntotal == 0:
            return []
        
        try:
            if len(query_vector) != self.dimension:
                return []
            
            query_np = np.array([query_vector], dtype=np.float32)
            faiss.normalize_L2(query_np)
            
            k_actual = min(k, self.index.ntotal)
            scores, indices = self.index.search(query_np, k_actual)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                
                vec_id = None
                if idx < len(self.position_to_id):
                    vec_id = self.position_to_id[idx]
                
                if vec_id:
                    results.append({
                        'id': vec_id,
                        'score': float(score)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas del índice."""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_path': str(self.index_path)
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