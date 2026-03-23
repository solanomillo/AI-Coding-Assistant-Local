#!/usr/bin/env python
"""
Script para probar la generación de embeddings con Gemini.
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from infrastructure.embeddings.gemini_embedding import GeminiEmbedding

# Configurar logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_embedding():
    """Prueba la generación de embedding."""
    print("=" * 60)
    print("PRUEBA DE EMBEDDING CON GEMINI")
    print("=" * 60)
    
    try:
        # Inicializar servicio
        print("\n1. Inicializando GeminiEmbedding...")
        emb = GeminiEmbedding()
        print(f"   ✅ Servicio inicializado")
        print(f"   📐 Dimensión esperada: {emb.get_dimension()}")
        
        # Probar con texto corto
        print("\n2. Probando con texto corto...")
        texto = "¿Cuántos archivos tiene el repositorio?"
        print(f"   📝 Texto: '{texto}'")
        print(f"   📏 Longitud: {len(texto)} caracteres")
        
        vector = emb.generate_embedding(texto)
        print(f"   ✅ Embedding generado")
        print(f"   📐 Dimensión: {len(vector)}")
        print(f"   📊 Primeros 5 valores: {vector[:5]}")
        
        if len(vector) == 3072:
            print("   ✅ CORRECTO: Dimensión 3072")
        else:
            print(f"   ❌ ERROR: Dimensión {len(vector)} != 3072")
        
        # Probar con texto más largo
        print("\n3. Probando con texto más largo...")
        texto_largo = "Explica la función sumar que recibe dos parámetros a y b y retorna la suma de ambos"
        print(f"   📝 Texto: '{texto_largo[:50]}...'")
        print(f"   📏 Longitud: {len(texto_largo)} caracteres")
        
        vector2 = emb.generate_embedding(texto_largo)
        print(f"   ✅ Embedding generado")
        print(f"   📐 Dimensión: {len(vector2)}")
        
        if len(vector2) == 3072:
            print("   ✅ CORRECTO: Dimensión 3072")
        else:
            print(f"   ❌ ERROR: Dimensión {len(vector2)} != 3072")
        
        # Probar múltiples
        print("\n4. Probando batch...")
        textos = [
            "¿Qué hace la función main?",
            "Explica la clase Calculadora",
            "Revisa el código por errores"
        ]
        
        vectors = emb.generate_embeddings_batch(textos)
        print(f"   ✅ Generados {len(vectors)} embeddings")
        for i, v in enumerate(vectors):
            print(f"   - Texto {i+1}: {len(v)} dimensiones - {'✅' if len(v) == 3072 else '❌'}")
        
        print("\n" + "=" * 60)
        print("PRUEBA COMPLETADA")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_embedding()
    sys.exit(0 if success else 1)