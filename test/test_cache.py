"""
Pruebas unitarias para el servicio de caché.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import time

from application.services.cache_service import CacheService


class TestCacheService(unittest.TestCase):
    """
    Pruebas para CacheService.
    """
    
    def setUp(self):
        """Configuración antes de cada prueba."""
        self.test_dir = tempfile.mkdtemp()
        self.cache = CacheService(
            cache_dir=Path(self.test_dir) / "cache",
            max_size_mb=1  # 1 MB para pruebas
        )
    
    def tearDown(self):
        """Limpieza después de cada prueba."""
        shutil.rmtree(self.test_dir)
    
    def test_put_and_get(self):
        """Prueba guardar y recuperar archivo."""
        content = b"test content"
        
        # Guardar
        cached_path = self.cache.put(1, "test.py", content)
        
        # Verificar que existe
        self.assertTrue(cached_path.exists())
        
        # Recuperar
        result = self.cache.get(1, "test.py")
        self.assertIsNotNone(result)
        self.assertEqual(result.read_bytes(), content)
    
    def test_cache_miss(self):
        """Prueba que cache miss retorne None."""
        result = self.cache.get(999, "nonexistent.py")
        self.assertIsNone(result)
    
    def test_text_operations(self):
        """Prueba operaciones con texto."""
        content = "print('hello world')"
        
        # Guardar texto
        self.cache.put_text(1, "main.py", content)
        
        # Recuperar texto
        result = self.cache.get_text(1, "main.py")
        self.assertEqual(result, content)
    
    def test_cleanup_when_full(self):
        """Prueba que se limpie cuando excede tamaño."""
        # Crear archivos hasta llenar caché
        for i in range(10):
            content = b"x" * 200 * 1024  # 200 KB
            self.cache.put(1, f"file{i}.py", content)
            time.sleep(0.1)  # Para diferentes last_access
        
        # Verificar que no excede tamaño máximo
        stats = self.cache.get_stats()
        self.assertLessEqual(stats['total_size_mb'], 1.1)  # Margen pequeño
    
    def test_clear_repository(self):
        """Prueba limpiar repositorio específico."""
        # Guardar archivos de dos repositorios
        self.cache.put(1, "repo1/file1.py", b"content1")
        self.cache.put(1, "repo1/file2.py", b"content2")
        self.cache.put(2, "repo2/file1.py", b"content3")
        
        # Limpiar repo 1
        self.cache.clear_repository(1)
        
        # Verificar
        self.assertIsNone(self.cache.get(1, "repo1/file1.py"))
        self.assertIsNone(self.cache.get(1, "repo1/file2.py"))
        self.assertIsNotNone(self.cache.get(2, "repo2/file1.py"))
    
    def test_contains(self):
        """Prueba operador contains."""
        self.cache.put(1, "test.py", b"content")
        
        self.assertIn((1, "test.py"), self.cache)
        self.assertNotIn((1, "other.py"), self.cache)
        self.assertNotIn((2, "test.py"), self.cache)
    
    def test_stats(self):
        """Prueba estadísticas."""
        # Guardar algunos archivos
        self.cache.put(1, "a.py", b"a" * 1000)
        self.cache.put(1, "b.py", b"b" * 2000)
        
        # Acceder varias veces
        self.cache.get(1, "a.py")
        self.cache.get(1, "a.py")
        self.cache.get(1, "b.py")
        
        stats = self.cache.get_stats()
        
        self.assertEqual(stats['total_files'], 2)
        self.assertGreater(stats['total_size_mb'], 0)
        self.assertEqual(stats['total_hits'], 3)
        self.assertEqual(len(stats['most_accessed']), 2)
    
    def test_prefetch(self):
        """Prueba precarga de archivos."""
        # Crear archivos temporales
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            
            # Crear algunos archivos
            (base / "src").mkdir()
            (base / "src/main.py").write_text("print('main')")
            (base / "src/utils.py").write_text("def helper(): pass")
            
            # Precargar
            file_paths = ["src/main.py", "src/utils.py"]
            self.cache.prefetch(1, file_paths, base)
            
            # Verificar
            self.assertIn((1, "src/main.py"), self.cache)
            self.assertIn((1, "src/utils.py"), self.cache)


if __name__ == '__main__':
    unittest.main()