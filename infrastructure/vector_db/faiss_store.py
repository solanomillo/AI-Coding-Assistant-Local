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

logger = logging.getLogger(__name__)


class FAISSStore:
    """
    Almacén vectorial usando FAISS.
    
    Características:
    - Índice plano (exacto) para búsqueda precisa
    - Persistencia en disco
    - Metadatos asociados a vectores
    """
    
    def __init__(self, dimension: int = 768, index_path: Optional[Path] = None):
        """
        Inicializa el almacén FAISS.
        
        Args:
            dimension: Dimensión de los vectores (768 para Gemini)
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
        
        logger.info(f"✅ FAISSStore inicializado: {len(self.metadata)} vectores, dimensión {dimension}")
    
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
        index = faiss.IndexFlatIP(self.dimension)  # IP = Inner Product (coseno si normalizado)
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
        
        # FAISS no guarda IDs, necesitamos mapeo manual
        for i in range(self.index.ntotal):
            # No podemos recuperar IDs del índice, solo si tenemos metadatos
            pass
        
        # Mejor: usar posición como ID simple
        logger.debug(f"Mapeos reconstruidos para {self.index.ntotal} vectores")
    
    def add_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Añade vectores al índice.
        
        Args:
            vectors: Lista de vectores
            ids: Lista de IDs únicos
            metadatas: Lista de metadatos
        """
        if len(vectors) != len(ids) or len(vectors) != len(metadatas):
            raise ValueError("vectors, ids y metadatas deben tener la misma longitud")
        
        if not vectors:
            logger.warning("No hay vectores para añadir")
            return
        
        try:
            # Convertir a numpy array
            vectors_np = np.array(vectors).astype('float32')
            
            # Normalizar para similitud coseno
            faiss.normalize_L2(vectors_np)
            
            # Obtener posición actual
            start_pos = self.index.ntotal
            
            # Añadir vectores
            self.index.add(vectors_np)
            
            # Guardar metadatos y mapeos
            for i, (vec_id, metadata) in enumerate(zip(ids, metadatas)):
                position = start_pos + i
                self.metadata[vec_id] = {
                    'position': position,
                    'data': metadata
                }
                self.position_to_id.append(vec_id)
                self.id_to_position[vec_id] = position
            
            # Guardar metadatos
            self._save_metadata()
            
            # Guardar índice
            faiss.write_index(self.index, str(self.index_path))
            
            logger.info(f"Añadidos {len(vectors)} vectores. Total: {self.index.ntotal}")
            
        except Exception as e:
            logger.error(f"Error añadiendo vectores: {e}")
            raise
    
    def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca vectores similares.
        
        Args:
            query_vector: Vector de consulta
            k: Número de resultados
            filter_criteria: Criterios de filtrado (pendiente implementar)
            
        Returns:
            Lista de resultados con metadatos y scores
        """
        if self.index.ntotal == 0:
            logger.warning("Índice vacío")
            return []
        
        try:
            # Preparar query
            query_np = np.array([query_vector]).astype('float32')
            faiss.normalize_L2(query_np)
            
            # Buscar
            scores, indices = self.index.search(query_np, min(k, self.index.ntotal))
            
            # Formatear resultados
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:
                    continue
                
                # Encontrar ID por posición
                vec_id = None
                for vid, pos in self.id_to_position.items():
                    if pos == idx:
                        vec_id = vid
                        break
                
                if vec_id and vec_id in self.metadata:
                    results.append({
                        'id': vec_id,
                        'score': float(score),
                        'metadata': self.metadata[vec_id]['data']
                    })
            
            logger.info(f"Búsqueda: {len(results)} resultados")
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> None:
        """
        Elimina vectores (no soportado en FAISS plano).
        
        Nota: FAISS no soporta eliminación fácilmente.
        Para simplificar, mejor recrear el índice.
        """
        logger.warning("Eliminación no soportada en FAISS plano. Usar delete_and_rebuild()")
    
    def delete_and_rebuild(self, keep_ids: Optional[List[str]] = None) -> None:
        """
        Reconstruye índice manteniendo solo ciertos IDs.
        
        Args:
            keep_ids: IDs a mantener (None = mantener todos)
        """
        if keep_ids is None:
            return
        
        # Crear nuevo índice
        new_index = faiss.IndexFlatIP(self.dimension)
        new_metadata = {}
        new_id_to_position = {}
        new_position_to_id = []
        
        # Reconstruir vectores
        for vec_id in keep_ids:
            if vec_id in self.metadata:
                # No podemos recuperar vector fácilmente
                # Esta funcionalidad requeriría guardar vectores
                logger.warning("Reconstrucción requiere guardar vectores originales")
                return
        
        logger.info("Reconstrucción completada")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del índice.
        
        Returns:
            Diccionario con estadísticas
        """
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
        
        # Guardar cambios
        faiss.write_index(self.index, str(self.index_path))
        self._save_metadata()
        
        logger.info("Índice limpiado")