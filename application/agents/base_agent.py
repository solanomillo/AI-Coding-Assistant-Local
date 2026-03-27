"""
Clase base para todos los agentes.
Define la interfaz común y funcionalidades compartidas.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import traceback

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Clase base abstracta para todos los agentes.
    """
    
    # Extensiones soportadas para búsqueda de archivos
    SUPPORTED_EXTENSIONS = {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.htm',
        '.css', '.scss', '.sass', '.json', '.yaml', '.yml',
        '.sql', '.sh', '.bash', '.go', '.rs', '.java',
        '.cpp', '.c', '.h', '.hpp', '.rb', '.php', '.vue',
        '.md', '.txt', '.rst'
    }
    
    def __init__(self, name: str, description: str):
        """
        Inicializa el agente base.
        
        Args:
            name: Nombre del agente
            description: Descripción de su función
        """
        self.name = name
        self.description = description
        self.llm = None
        self.vector_store = None
        self.repo_context = None
        self.embedding_service = None
        self.cache_service = None
        self.repo_path = None
        logger.info(f"Agente '{name}' inicializado: {description}")
    
    def set_llm(self, llm) -> None:
        """Establece el cliente LLM para el agente."""
        self.llm = llm
        logger.debug(f"LLM configurado para agente {self.name}")
    
    def set_vector_store(self, vector_store) -> None:
        """Establece el vector store para búsqueda de contexto."""
        self.vector_store = vector_store
        logger.debug(f"Vector store configurado para agente {self.name}")
    
    def set_embedding_service(self, embedding_service) -> None:
        """Establece el servicio de embeddings."""
        self.embedding_service = embedding_service
        logger.debug(f"Embedding service configurado para agente {self.name}")
    
    def set_cache_service(self, cache_service) -> None:
        """Establece el servicio de caché para recuperar fragmentos completos."""
        self.cache_service = cache_service
        logger.debug(f"Cache service configurado para agente {self.name}")
    
    def set_repo_path(self, repo_path: Path) -> None:
        """Establece la ruta del repositorio para leer archivos completos."""
        self.repo_path = repo_path
        logger.debug(f"Repo path configurado para agente {self.name}")
    
    def set_repo_context(self, repo_context: Dict[str, Any]) -> None:
        """Establece el contexto del repositorio."""
        self.repo_context = repo_context
        logger.debug(f"Contexto de repositorio configurado para agente {self.name}")
    
    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """Determina si este agente puede manejar la consulta."""
        pass
    
    @abstractmethod
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Procesa la consulta y genera respuesta."""
        pass
    
    def _extract_file_name(self, query: str) -> Optional[str]:
        """
        Extrae el nombre de un archivo de la consulta.
        Soporta múltiples extensiones: .py, .js, .html, .css, etc.
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Nombre del archivo o None
        """
        query_lower = query.lower()
        
        # Construir patrones para todas las extensiones soportadas
        extensions_pattern = '|'.join([re.escape(ext) for ext in self.SUPPORTED_EXTENSIONS])
        
        # Patrones para detectar archivos
        patterns = [
            rf'archivo\s+([a-zA-Z0-9_\-\.]+(?:{extensions_pattern}))',
            rf'archivo\s+([a-zA-Z0-9_\-\.]+)',
            rf'([a-zA-Z0-9_\-\.]+(?:{extensions_pattern}))',
            rf'en\s+([a-zA-Z0-9_\-\.]+(?:{extensions_pattern}))',
            rf'de\s+([a-zA-Z0-9_\-\.]+(?:{extensions_pattern}))',
            rf'el\s+archivo\s+([a-zA-Z0-9_\-\.]+(?:{extensions_pattern}))',
            rf'la\s+función\s+en\s+([a-zA-Z0-9_\-\.]+(?:{extensions_pattern}))',
            rf'el\s+código\s+en\s+([a-zA-Z0-9_\-\.]+(?:{extensions_pattern}))',
            rf'el\s+estilo\s+en\s+([a-zA-Z0-9_\-\.]+(?:{extensions_pattern}))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                file_name = match.group(1)
                logger.debug(f"Archivo detectado: {file_name}")
                return file_name
        
        return None
    
    def _get_full_file_content(self, file_name: str) -> Optional[str]:
        """
        Obtiene el contenido completo de un archivo.
        Soporta búsqueda exacta y parcial del nombre.
        
        Args:
            file_name: Nombre del archivo
            
        Returns:
            Contenido completo del archivo o None
        """
        if not self.repo_path:
            logger.warning("Repo path no configurado")
            return None
        
        logger.debug(f"Buscando archivo: {file_name} en {self.repo_path}")
        
        # Buscar el archivo exacto
        for path in self.repo_path.rglob(file_name):
            if path.is_file():
                try:
                    content = path.read_text(encoding='utf-8', errors='ignore')
                    ext = path.suffix.lower()
                    logger.info(f"Archivo completo recuperado: {file_name} ({len(content)} caracteres, {ext})")
                    return content
                except Exception as e:
                    logger.error(f"Error leyendo archivo {file_name}: {e}")
                    return None
        
        # Si no se encuentra exacto, buscar por nombre base (sin ruta)
        base_name = Path(file_name).name
        for path in self.repo_path.rglob(base_name):
            if path.is_file():
                try:
                    content = path.read_text(encoding='utf-8', errors='ignore')
                    logger.info(f"Archivo encontrado por nombre base: {path.name} ({len(content)} caracteres)")
                    return content
                except Exception as e:
                    logger.error(f"Error leyendo archivo {path.name}: {e}")
                    continue
        
        logger.warning(f"No se encontró el archivo: {file_name}")
        return None
    
    def _get_file_language(self, file_name: str) -> str:
        """
        Determina el lenguaje de un archivo por su extensión.
        
        Args:
            file_name: Nombre del archivo
            
        Returns:
            Nombre del lenguaje
        """
        ext = Path(file_name).suffix.lower()
        
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.jsx': 'React JSX',
            '.ts': 'TypeScript',
            '.tsx': 'React TypeScript',
            '.html': 'HTML',
            '.htm': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.sass': 'SASS',
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.sql': 'SQL',
            '.sh': 'Shell Script',
            '.bash': 'Bash Script',
            '.go': 'Go',
            '.rs': 'Rust',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C/C++ Header',
            '.hpp': 'C++ Header',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.vue': 'Vue.js',
            '.md': 'Markdown',
            '.txt': 'Texto Plano',
            '.rst': 'reStructuredText'
        }
        
        return language_map.get(ext, 'Desconocido')
    
    def _retrieve_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Recupera contexto relevante.
        Prioriza archivos completos si se mencionan en la consulta.
        """
        # Intentar extraer nombre de archivo de la consulta
        file_name = self._extract_file_name(query)
        
        if file_name:
            logger.info(f"📄 Archivo específico detectado: {file_name}")
            full_content = self._get_full_file_content(file_name)
            if full_content:
                language = self._get_file_language(file_name)
                logger.info(f"✅ Archivo completo recuperado: {file_name} ({len(full_content)} caracteres, {language})")
                return [{
                    'id': f"full:{file_name}",
                    'file': file_name,
                    'content': full_content,
                    'language': language,
                    'score': 1.0,
                    'is_full_file': True
                }]
            else:
                logger.warning(f"❌ No se pudo recuperar archivo: {file_name}")
        
        # DETECTAR CONSULTAS GENERALES SOBRE EL REPOSITORIO
        query_lower = query.lower()
        general_queries = [
            'qué hace', 'de qué trata', 'resumen', 'qué es', 'explica el repositorio',
            'qué hace este proyecto', 'descripción', 'funcionalidad', 'propósito',
            'qué contiene', 'qué archivos', 'estructura'
        ]
        
        is_general_query = any(phrase in query_lower for phrase in general_queries)
        
        # Si es una consulta general, recuperar TODOS los archivos del repositorio
        if is_general_query and self.repo_path:
            logger.info("🔍 Consulta general detectada - recuperando archivos relevantes")
            
            # Definir extensiones a incluir (priorizar HTML, CSS, JS)
            extensions_to_include = {
                '.html': 0,
                '.css': 1,
                '.js': 2,
                '.py': 3
            }
            
            all_files = []
            
            for ext, priority in extensions_to_include.items():
                for file_path in self.repo_path.rglob(f"*{ext}"):
                    # Ignorar archivos en directorios comunes
                    if any(ignore in file_path.parts for ignore in ['__pycache__', 'node_modules', 'venv', '.git']):
                        continue
                    
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        if content.strip():
                            # Limitar tamaño para no saturar el contexto
                            if len(content) > 2000:
                                content = content[:2000] + "\n... (contenido truncado)"
                            
                            all_files.append({
                                'file': file_path.name,
                                'path': str(file_path.relative_to(self.repo_path)),
                                'content': content,
                                'language': self._get_file_language(file_path.name),
                                'priority': priority,
                                'size': len(content)
                            })
                            logger.debug(f"Archivo encontrado: {file_path.name} ({len(content)} caracteres)")
                            
                    except Exception as e:
                        logger.debug(f"Error leyendo {file_path}: {e}")
            
            if all_files:
                # Ordenar por prioridad (HTML primero, luego CSS, luego JS)
                all_files.sort(key=lambda x: (x['priority'], -x['size']))
                
                logger.info(f"📁 Recuperados {len(all_files)} archivos para consulta general")
                for f in all_files:
                    logger.info(f"  - {f['path']} ({f['language']}) - {f['size']} caracteres")
                
                # Construir contexto con todos los archivos
                combined_content = []
                for f in all_files[:8]:  # Limitar a 8 archivos
                    combined_content.append(
                        f"[Archivo: {f['path']} ({f['language']})]\n"
                        f"```{f['language'].lower()}\n"
                        f"{f['content']}\n"
                        f"```\n"
                    )
                
                return [{
                    'id': "full:repository_summary",
                    'file': "resumen_repositorio",
                    'content': "\n---\n".join(combined_content),
                    'language': "multi",
                    'score': 1.0,
                    'is_full_file': True,
                    'is_repository_summary': True
                }]
        
        # Si no es consulta general, usar búsqueda vectorial
        if not self.vector_store:
            logger.warning(f"Vector store no disponible para agente {self.name}")
            return []
        
        if not self.embedding_service:
            logger.warning(f"Embedding service no disponible para agente {self.name}")
            return []
        
        if not self.cache_service:
            logger.warning(f"Cache service no disponible para agente {self.name}")
            return []
        
        try:
            logger.debug(f"Generando embedding para consulta: {query[:50]}...")
            query_vector = self.embedding_service.generate_embedding(query)
            
            if len(query_vector) != self.embedding_service.get_dimension():
                logger.error(f"Dimensión incorrecta: {len(query_vector)}")
                return []
            
            logger.debug(f"Buscando en FAISS con k={k}...")
            results = self.vector_store.search(query_vector, k=k)
            
            if not results:
                logger.debug("No se encontraron resultados en FAISS")
                return []
            
            logger.info(f"FAISS encontró {len(results)} resultados")
            
            # Recuperar fragmentos completos del caché
            fragments = []
            for r in results:
                chunk_id = r['id']
                content = self.cache_service.get_chunk(chunk_id)
                if content:
                    parts = chunk_id.split(':')
                    file_name = parts[1] if len(parts) > 1 else 'desconocido'
                    fragments.append({
                        'id': chunk_id,
                        'file': file_name,
                        'content': content,
                        'score': r['score']
                    })
            
            logger.info(f"Recuperados {len(fragments)} fragmentos del caché")
            
            if not fragments:
                logger.warning("No se recuperaron fragmentos del caché")
                return []
            
            # Intentar obtener archivo completo si hay suficientes fragmentos
            if len(fragments) >= 2:
                fragments_by_file = {}
                for f in fragments:
                    file = f['file']
                    if file not in fragments_by_file:
                        fragments_by_file[file] = []
                    fragments_by_file[file].append(f)
                
                best_file = max(fragments_by_file.keys(), key=lambda x: len(fragments_by_file[x]))
                if len(fragments_by_file[best_file]) >= 2:
                    full_content = self._get_full_file_content(best_file)
                    if full_content:
                        language = self._get_file_language(best_file)
                        logger.info(f"📄 Archivo completo recuperado por fragmentos: {best_file} ({len(full_content)} caracteres)")
                        return [{
                            'id': f"full:{best_file}",
                            'file': best_file,
                            'content': full_content,
                            'language': language,
                            'score': 1.0,
                            'is_full_file': True
                        }]
            
            logger.debug(f"Retornando {len(fragments)} fragmentos para {self.name}")
            return fragments[:k]
            
        except Exception as e:
            logger.error(f"Error recuperando contexto: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _build_context_text(self, fragments: List[Dict[str, Any]]) -> str:
        """
        Construye texto de contexto a partir de fragmentos.
        
        Args:
            fragments: Lista de fragmentos recuperados
            
        Returns:
            Texto de contexto formateado
        """
        context_parts = []
        
        for i, f in enumerate(fragments):
            language_info = f.get('language', 'código')
            
            if f.get('is_full_file'):
                context_parts.append(
                    f"[ARCHIVO COMPLETO: {f['file']} ({language_info})]\n"
                    f"```{language_info.lower()}\n"
                    f"{f['content']}\n"
                    f"```\n"
                )
            else:
                context_parts.append(
                    f"[Fragmento {i+1} - Archivo: {f['file']}]\n"
                    f"```{language_info.lower()}\n"
                    f"{f['content']}\n"
                    f"```\n"
                )
        
        full_context = "\n---\n".join(context_parts)
        logger.info(f"Contexto construido: {len(full_context)} caracteres, {len(fragments)} fragmentos")
        
        # Si el contexto es muy grande, truncar (pero mantener archivo completo)
        if len(full_context) > 15000:
            logger.warning(f"Contexto muy grande ({len(full_context)} caracteres), truncando a 15000...")
            full_context = full_context[:15000] + "\n... (contexto truncado)"
        
        return full_context
    
    def _build_prompt(self, query: str, context: str, instructions: str) -> str:
        """
        Construye prompt para el LLM.
        
        Args:
            query: Consulta del usuario
            context: Contexto recuperado
            instructions: Instrucciones específicas del agente
            
        Returns:
            Prompt completo
        """
        repo_info = ""
        if self.repo_context:
            repo_info = f"Repositorio: {self.repo_context.get('name', 'desconocido')}\n"
        
        prompt = f"""{instructions}

{repo_info}
CONTEXTO DEL CÓDIGO:
{context}

CONSULTA DEL USUARIO:
{query}

INSTRUCCIONES ADICIONALES:
- Analiza TODO el código proporcionado en el contexto
- Si se proporciona un archivo completo, analiza todas las funciones, clases, estilos o elementos
- Identifica el lenguaje de programación de cada archivo
- Responde basándote ÚNICAMENTE en el código que ves
- Si el código no está presente, indícalo claramente
- Sé técnico y preciso
- Usa formato de código con ``` y especifica el lenguaje
- Responde en español

RESPUESTA:"""
        
        logger.info(f"Prompt construido: {len(prompt)} caracteres")
        return prompt