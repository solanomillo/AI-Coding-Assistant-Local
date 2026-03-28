"""
Clase base para todos los agentes.
Define la interfaz común y funcionalidades compartidas.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Clase base abstracta para todos los agentes.
    """
    
    SUPPORTED_EXTENSIONS = {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.htm',
        '.css', '.scss', '.sass', '.json', '.yaml', '.yml',
        '.sql', '.sh', '.bash', '.go', '.rs', '.java',
        '.cpp', '.c', '.h', '.hpp', '.rb', '.php', '.vue',
        '.md', '.txt', '.rst'
    }
    
    # Palabras clave para consultas generales
    GENERAL_QUERY_KEYWORDS = [
        'qué hace', 'de qué trata', 'resumen', 'qué es', 'explica el repositorio',
        'qué hace este proyecto', 'descripción', 'funcionalidad', 'propósito',
        'qué contiene', 'qué archivos', 'estructura', 'que hace', 'que es'
    ]
    
    def __init__(self, name: str, description: str):
        """
        Inicializa el agente base.
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
        self.llm = llm
        logger.debug(f"LLM configurado para agente {self.name}")
    
    def set_vector_store(self, vector_store) -> None:
        self.vector_store = vector_store
        logger.debug(f"Vector store configurado para agente {self.name}")
    
    def set_embedding_service(self, embedding_service) -> None:
        self.embedding_service = embedding_service
        logger.debug(f"Embedding service configurado para agente {self.name}")
    
    def set_cache_service(self, cache_service) -> None:
        self.cache_service = cache_service
        logger.debug(f"Cache service configurado para agente {self.name}")
    
    def set_repo_path(self, repo_path: Path) -> None:
        self.repo_path = repo_path
        logger.debug(f"Repo path configurado para agente {self.name}")
    
    def set_repo_context(self, repo_context: Dict[str, Any]) -> None:
        self.repo_context = repo_context
        logger.debug(f"Contexto de repositorio configurado para agente {self.name}")
    
    @abstractmethod
    def can_handle(self, query: str) -> bool:
        pass
    
    @abstractmethod
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        pass
    
    def _is_general_query(self, query: str) -> bool:
        """Determina si la consulta es general sobre el repositorio."""
        query_lower = query.lower()
        return any(phrase in query_lower for phrase in self.GENERAL_QUERY_KEYWORDS)
    
    def _extract_file_name(self, query: str) -> Optional[str]:
        """Extrae el nombre de un archivo de la consulta."""
        query_lower = query.lower()
        extensions_pattern = '|'.join([re.escape(ext) for ext in self.SUPPORTED_EXTENSIONS])
        
        patterns = [
            rf'archivo\s+([a-zA-Z0-9_\-\.]+(?:{extensions_pattern}))',
            rf'archivo\s+([a-zA-Z0-9_\-\.]+)',
            rf'([a-zA-Z0-9_\-\.]+(?:{extensions_pattern}))',
            rf'en\s+([a-zA-Z0-9_\-\.]+(?:{extensions_pattern}))',
            rf'de\s+([a-zA-Z0-9_\-\.]+(?:{extensions_pattern}))',
            rf'el\s+archivo\s+([a-zA-Z0-9_\-\.]+(?:{extensions_pattern}))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                file_name = match.group(1)
                logger.debug(f"Archivo detectado: {file_name}")
                return file_name
        
        return None
    
    def _get_full_file_content(self, file_name: str) -> Optional[str]:
        """Obtiene el contenido completo de un archivo."""
        if not self.repo_path:
            logger.warning("Repo path no configurado")
            return None
        
        logger.debug(f"Buscando archivo: {file_name} en {self.repo_path}")
        
        for path in self.repo_path.rglob(file_name):
            if path.is_file():
                try:
                    content = path.read_text(encoding='utf-8', errors='ignore')
                    logger.info(f"Archivo completo recuperado: {file_name} ({len(content)} caracteres)")
                    return content
                except Exception as e:
                    logger.error(f"Error leyendo archivo {file_name}: {e}")
                    return None
        
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
        """Determina el lenguaje de un archivo por su extensión."""
        ext = Path(file_name).suffix.lower()
        
        language_map = {
            '.py': 'Python', '.js': 'JavaScript', '.jsx': 'React JSX',
            '.ts': 'TypeScript', '.tsx': 'React TypeScript',
            '.html': 'HTML', '.htm': 'HTML', '.css': 'CSS',
            '.scss': 'SCSS', '.sass': 'SASS', '.json': 'JSON',
            '.yaml': 'YAML', '.yml': 'YAML', '.sql': 'SQL',
            '.sh': 'Shell Script', '.bash': 'Bash Script', '.go': 'Go',
            '.rs': 'Rust', '.java': 'Java', '.cpp': 'C++', '.c': 'C',
            '.h': 'C/C++ Header', '.hpp': 'C++ Header', '.rb': 'Ruby',
            '.php': 'PHP', '.vue': 'Vue.js', '.md': 'Markdown',
            '.txt': 'Texto Plano', '.rst': 'reStructuredText'
        }
        
        return language_map.get(ext, 'Desconocido')
    
    def _extract_file_summary(self, file_path: Path, content: str, language: str) -> Dict[str, Any]:
        """
        Extrae un resumen compacto de un archivo.
        Optimizado para minimizar tokens en consultas generales.
        """
        summary = {
            'name': file_path.name,
            'path': str(file_path.relative_to(self.repo_path)) if self.repo_path else file_path.name,
            'language': language,
            'size': len(content),
            'line_count': content.count('\n') + 1,
            'purpose': self._infer_file_purpose(file_path.name, language),
            'key_elements': []
        }
        
        # HTML: extraer título, meta tags, estructura principal
        if language == 'HTML':
            title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
            if title_match:
                summary['title'] = title_match.group(1).strip()
            
            script_count = len(re.findall(r'<script', content, re.IGNORECASE))
            link_count = len(re.findall(r'<link[^>]*stylesheet', content, re.IGNORECASE))
            summary['key_elements'] = [f"{script_count} scripts", f"{link_count} estilos"]
            
            main_tags = []
            for tag in ['h1', 'h2', 'main', 'header', 'footer', 'nav']:
                if re.search(f'<{tag}[^>]*>', content, re.IGNORECASE):
                    main_tags.append(tag)
            if main_tags:
                summary['structure'] = main_tags
        
        # CSS: extraer variables, clases principales, media queries
        elif language == 'CSS':
            variables = re.findall(r'--[a-zA-Z0-9_-]+:', content)
            if variables:
                summary['variables_count'] = len(set(variables))
            
            classes = re.findall(r'\.([a-zA-Z0-9_-]+)\s*\{', content)
            if classes:
                summary['classes_count'] = len(set(classes))
                summary['main_classes'] = list(set(classes))[:5]
            
            media_count = len(re.findall(r'@media', content, re.IGNORECASE))
            if media_count:
                summary['media_queries'] = media_count
        
        # JavaScript: extraer funciones, clases, eventos
        elif language == 'JavaScript':
            functions = re.findall(r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(', content)
            arrow_funcs = re.findall(r'(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*\([^)]*\)\s*=>', content)
            all_funcs = list(set(functions + arrow_funcs))
            if all_funcs:
                summary['functions'] = all_funcs[:8]
                summary['functions_count'] = len(all_funcs)
            
            classes = re.findall(r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', content)
            if classes:
                summary['classes'] = classes[:5]
                summary['classes_count'] = len(classes)
            
            event_listeners = re.findall(r'\.(addEventListener|onclick|onload|onchange)\s*\(', content)
            if event_listeners:
                summary['events'] = list(set(event_listeners))
        
        # Python: extraer funciones, clases, imports
        elif language == 'Python':
            functions = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', content)
            if functions:
                summary['functions'] = functions[:8]
                summary['functions_count'] = len(functions)
            
            classes = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]', content)
            if classes:
                summary['classes'] = classes[:5]
                summary['classes_count'] = len(classes)
            
            imports = re.findall(r'^(?:from\s+(\S+)\s+import|import\s+(\S+))', content, re.MULTILINE)
            if imports:
                summary['imports_count'] = len(imports)
        
        return summary
    
    def _infer_file_purpose(self, filename: str, language: str) -> str:
        """Infiera el propósito del archivo por su nombre."""
        name_lower = filename.lower()
        
        purpose_map = {
            'index': 'Pagina principal de la aplicacion',
            'main': 'Punto de entrada principal',
            'app': 'Logica principal de la aplicacion',
            'style': 'Estilos visuales de la interfaz',
            'styles': 'Estilos visuales de la interfaz',
            'script': 'Funcionalidad JavaScript',
            'utils': 'Utilidades y funciones auxiliares',
            'helpers': 'Funciones de ayuda',
            'config': 'Configuracion del proyecto',
            'routes': 'Definicion de rutas',
            'models': 'Modelos de datos',
            'views': 'Vistas de la aplicacion',
            'controllers': 'Controladores de la aplicacion',
            'service': 'Servicios de la aplicacion'
        }
        
        for key, purpose in purpose_map.items():
            if key in name_lower:
                return purpose
        
        if language == 'HTML':
            return 'Documento HTML que define la estructura de la pagina'
        elif language == 'CSS':
            return 'Hoja de estilos para el diseno visual'
        elif language == 'JavaScript':
            return 'Codigo JavaScript con funcionalidad interactiva'
        elif language == 'Python':
            return 'Modulo Python con logica de negocio'
        
        return f'Archivo {language}'
    
    def _extract_file_summary(self, file_path: Path, content: str, language: str) -> Dict[str, Any]:
        """
        Extrae un resumen compacto de un archivo.
        Optimizado para minimizar tokens en consultas generales.
        """
        summary = {
            'name': file_path.name,
            'path': str(file_path.relative_to(self.repo_path)) if self.repo_path else file_path.name,
            'language': language,
            'size': len(content),
            'line_count': content.count('\n') + 1,
            'purpose': self._infer_file_purpose(file_path.name, language),
            'key_elements': []
        }
        
        # HTML: extraer título, meta tags, estructura principal
        if language == 'HTML':
            title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
            if title_match:
                summary['title'] = title_match.group(1).strip()
            
            script_count = len(re.findall(r'<script', content, re.IGNORECASE))
            link_count = len(re.findall(r'<link[^>]*stylesheet', content, re.IGNORECASE))
            summary['key_elements'] = [f"{script_count} scripts", f"{link_count} estilos"]
            
            main_tags = []
            for tag in ['h1', 'h2', 'main', 'header', 'footer', 'nav']:
                if re.search(f'<{tag}[^>]*>', content, re.IGNORECASE):
                    main_tags.append(tag)
            if main_tags:
                summary['structure'] = main_tags
        
        # CSS: extraer variables, clases principales, media queries
        elif language == 'CSS':
            variables = re.findall(r'--[a-zA-Z0-9_-]+:', content)
            if variables:
                summary['variables_count'] = len(set(variables))
            
            classes = re.findall(r'\.([a-zA-Z0-9_-]+)\s*\{', content)
            if classes:
                summary['classes_count'] = len(set(classes))
                summary['main_classes'] = list(set(classes))[:5]
            
            media_count = len(re.findall(r'@media', content, re.IGNORECASE))
            if media_count:
                summary['media_queries'] = media_count
        
        # JavaScript: extraer funciones, clases, eventos (mejorado)
        elif language == 'JavaScript':
            # Funciones declaradas: function nombre() {}
            functions = re.findall(r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(', content)
            # Funciones flecha asignadas: const nombre = () => {}
            arrow_funcs = re.findall(r'(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*\([^)]*\)\s*=>', content)
            # Funciones flecha sin asignación: () => {}
            all_funcs = list(set(functions + arrow_funcs))
            if all_funcs:
                summary['functions'] = all_funcs[:8]
                summary['functions_count'] = len(all_funcs)
            
            # Clases
            classes = re.findall(r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', content)
            if classes:
                summary['classes'] = classes[:5]
                summary['classes_count'] = len(classes)
            
            # Variables globales (const, let, var a nivel raíz)
            variables = re.findall(r'^(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=', content, re.MULTILINE)
            if variables:
                summary['variables'] = variables[:5]
            
            # Event listeners
            event_listeners = re.findall(r'\.(addEventListener|onclick|onload|onchange|onsubmit)\s*\(', content)
            if event_listeners:
                summary['events'] = list(set(event_listeners))
            
            # Indica si hay lógica interactiva
            if 'functions' in summary or 'event_listeners' in locals():
                summary['has_interactivity'] = True
        
        # Python: extraer funciones, clases, imports
        elif language == 'Python':
            functions = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', content)
            if functions:
                summary['functions'] = functions[:8]
                summary['functions_count'] = len(functions)
            
            classes = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]', content)
            if classes:
                summary['classes'] = classes[:5]
                summary['classes_count'] = len(classes)
            
            imports = re.findall(r'^(?:from\s+(\S+)\s+import|import\s+(\S+))', content, re.MULTILINE)
            if imports:
                summary['imports_count'] = len(imports)
        
        return summary
    
    def _build_compact_context(self, repo_summary: Dict[str, Any]) -> str:
        """
        Construye un contexto compacto a partir del resumen del repositorio.
        Sin fragmentos de código, solo información estructurada.
        """
        context_parts = []
        
        context_parts.append(f"El repositorio contiene {repo_summary['total_files']} archivos:\n")
        
        for file in repo_summary['files'][:8]:
            # Información básica del archivo
            file_info = f"- **{file['path']}** ({file['language']}, {file['line_count']} lineas)"
            
            # Propósito del archivo
            if file.get('purpose'):
                file_info += f"\n  Proposito: {file['purpose']}"
            
            # Título (para HTML)
            if file.get('title'):
                file_info += f"\n  Titulo: {file['title']}"
            
            # Funciones principales
            if file.get('functions'):
                func_list = ', '.join(file['functions'][:5])
                file_info += f"\n  Funciones: {func_list}"
                if file.get('functions_count', 0) > 5:
                    file_info += f" (+{file['functions_count'] - 5} mas)"
            
            # Clases principales
            if file.get('classes'):
                class_list = ', '.join(file['classes'][:3])
                file_info += f"\n  Clases: {class_list}"
            
            # Clases CSS principales
            if file.get('main_classes'):
                file_info += f"\n  Clases CSS: {', '.join(file['main_classes'][:5])}"
            
            # Elementos clave (scripts, estilos)
            if file.get('key_elements'):
                file_info += f"\n  Elementos: {', '.join(file['key_elements'])}"
            
            context_parts.append(file_info)
        
        return "\n\n".join(context_parts)
    
    def _retrieve_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Recupera contexto relevante.
        """
        file_name = self._extract_file_name(query)
        
        # Caso 1: Archivo especifico mencionado
        if file_name:
            logger.info(f"Archivo especifico detectado: {file_name}")
            full_content = self._get_full_file_content(file_name)
            if full_content:
                language = self._get_file_language(file_name)
                return [{
                    'id': f"full:{file_name}",
                    'file': file_name,
                    'content': full_content,
                    'language': language,
                    'score': 1.0,
                    'is_full_file': True
                }]
        
        # Caso 2: Consulta general sobre el repositorio
        if self._is_general_query(query) and self.repo_path:
            logger.info("Consulta general detectada - generando resumen compacto")
            
            repo_summary = self._generate_repository_summary()
            
            if repo_summary['files']:
                compact_context = self._build_compact_context(repo_summary)
                logger.info(f"Resumen generado: {repo_summary['total_files']} archivos, {len(compact_context)} caracteres")
                
                return [{
                    'id': "full:repository_summary",
                    'file': "resumen_repositorio",
                    'content': compact_context,
                    'language': "multi",
                    'score': 1.0,
                    'is_full_file': True,
                    'is_repository_summary': True
                }]
        
        # Caso 3: Busqueda vectorial para consultas especificas
        if not self.vector_store or not self.embedding_service or not self.cache_service:
            logger.warning("Servicios no disponibles para busqueda vectorial")
            return []
        
        try:
            logger.debug(f"Generando embedding para consulta: {query[:50]}...")
            query_vector = self.embedding_service.generate_embedding(query)
            
            if len(query_vector) != self.embedding_service.get_dimension():
                logger.error(f"Dimensión incorrecta: {len(query_vector)}")
                return []
            
            results = self.vector_store.search(query_vector, k=k)
            
            if not results:
                logger.debug("No se encontraron resultados")
                return []
            
            logger.info(f"FAISS encontró {len(results)} resultados")
            
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
            
            logger.info(f"Recuperados {len(fragments)} fragmentos")
            return fragments[:k]
            
        except Exception as e:
            logger.error(f"Error recuperando contexto: {e}")
            return []
    
    def _build_context_text(self, fragments: List[Dict[str, Any]]) -> str:
        """
        Construye texto de contexto a partir de fragmentos.
        """
        context_parts = []
        
        for i, f in enumerate(fragments):
            language_info = f.get('language', 'codigo')
            
            if f.get('is_full_file'):
                if f.get('is_repository_summary'):
                    context_parts.append(
                        f"[RESUMEN DEL REPOSITORIO]\n"
                        f"{f['content']}\n"
                    )
                else:
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
        
        if len(full_context) > 15000:
            logger.warning(f"Contexto muy grande, truncando...")
            full_context = full_context[:15000] + "\n... (contexto truncado)"
        
        return full_context
    
    def _build_prompt(self, query: str, context: str, instructions: str) -> str:
        """
        Construye prompt para el LLM.
        """
        repo_info = ""
        if self.repo_context:
            repo_info = f"Repositorio: {self.repo_context.get('name', 'desconocido')}\n"
        
        prompt = f"""{instructions}

{repo_info}
CONTEXTO DEL CODIGO:
{context}

CONSULTA DEL USUARIO:
{query}

INSTRUCCIONES ADICIONALES:
- Analiza TODO el codigo proporcionado en el contexto
- Si se proporciona un resumen del repositorio, analiza todos los archivos incluidos
- Identifica el lenguaje de programacion de cada archivo
- Responde basandote UNICAMENTE en el codigo que ves
- Si el codigo no esta presente, indicarlo claramente
- Ser tecnico y preciso
- Usar formato de codigo con ``` y especificar el lenguaje
- Responder en español

RESPUESTA:"""
        
        logger.info(f"Prompt construido: {len(prompt)} caracteres")
        return prompt