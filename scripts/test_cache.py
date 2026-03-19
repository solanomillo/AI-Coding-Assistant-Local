#!/usr/bin/env python
"""
Script para probar el sistema de caché manualmente.
"""

import sys
from pathlib import Path
import time

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from application.services.cache_service import CacheService


def test_cache():
    """Prueba interactiva del caché."""
    print("=" * 50)
    print("Prueba del Sistema de Caché")
    print("=" * 50)
    
    # Inicializar caché
    cache = CacheService(max_size_mb=10)
    print(f"\n✅ Caché inicializado")
    print(f"   Directorio: {cache.cache_dir}")
    
    # Probar guardar texto
    print("\n📝 Probando guardar texto...")
    cache.put_text(1, "test.py", "print('hello world')")
    print("   ✅ Texto guardado")
    
    # Probar recuperar
    print("\n🔍 Probando recuperar...")
    content = cache.get_text(1, "test.py")
    print(f"   ✅ Contenido recuperado: {content}")
    
    # Mostrar estadísticas
    print("\n📊 Estadísticas del caché:")
    stats = cache.get_stats()
    for key, value in stats.items():
        if key != 'most_accessed':
            print(f"   {key}: {value}")
    
    # Mostrar archivos más accedidos
    if stats['most_accessed']:
        print("\n⭐ Archivos más accedidos:")
        for f in stats['most_accessed']:
            print(f"   - {f['path']} ({f['hits']} hits)")
    
    print("\n✅ Prueba completada")


if __name__ == "__main__":
    test_cache()