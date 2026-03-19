#!/usr/bin/env python
"""
Script para probar la extracción de ZIP sin Streamlit.
"""

import zipfile
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_zip(zip_path: str):
    """Prueba extracción de ZIP."""
    zip_path = Path(zip_path)
    
    if not zip_path.exists():
        logger.error(f"❌ ZIP no encontrado: {zip_path}")
        return
    
    logger.info(f"📦 Probando ZIP: {zip_path}")
    logger.info(f"📦 Tamaño: {zip_path.stat().st_size / 1024:.2f} KB")
    
    # Crear directorio de prueba
    test_dir = Path("test_extract")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Extraer
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(test_dir)
        
        logger.info("✅ ZIP extraído correctamente")
        
        # Listar contenido
        logger.info("📋 Contenido:")
        for item in test_dir.iterdir():
            logger.info(f"   - {item.name} {'📁' if item.is_dir() else '📄'}")
        
        # Buscar archivos Python
        py_files = list(test_dir.rglob("*.py"))
        logger.info(f"🐍 Archivos Python: {len(py_files)}")
        
        if py_files:
            logger.info("Primeros 5 archivos:")
            for f in py_files[:5]:
                logger.info(f"   - {f.relative_to(test_dir)}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    
    # Limpiar
    import shutil
    shutil.rmtree(test_dir)
    logger.info("🧹 Directorio de prueba eliminado")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_zip(sys.argv[1])
    else:
        print("Uso: python test_zip.py <ruta_al_zip>")