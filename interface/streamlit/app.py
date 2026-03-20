"""
Módulo de interfaz de usuario para AI Coding Assistant Local.
Actualizado con soporte multi-lenguaje y estadísticas de caché.
"""

import streamlit as st
import logging
from pathlib import Path
import tempfile
import os
from dotenv import load_dotenv, set_key
from datetime import datetime

from application.services.repo_service import RepositoryService
from application.services.rag_service import RAGService
from infrastructure.llm.gemini_llm import GeminiLLM

logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()
ENV_FILE = ".env"

# Configuración de página
st.set_page_config(
    page_title="🤖 AI Coding Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def save_env_var(key: str, value: str) -> bool:
    """Guarda variable en .env."""
    try:
        set_key(ENV_FILE, key, value)
        os.environ[key] = value
        return True
    except Exception as e:
        logger.error(f"Error guardando {key}: {e}")
        return False


def show_upload_section() -> None:
    """Sección de carga con soporte multi-lenguaje."""
    st.title("📤 Cargar Repositorio")
    
    # Inicializar servicio
    if 'repo_service' not in st.session_state:
        st.session_state.repo_service = RepositoryService()
    
    # Verificar API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ **API Key no configurada**")
        st.info("👉 Ve a la pestaña **Configuración** para agregar tu API key de Gemini")
        return
    
    # Mostrar lenguajes soportados
    with st.expander("🌐 Lenguajes Soportados", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🐍 Python**")
            st.markdown("**📜 JavaScript/TypeScript**")
            st.markdown("**🌐 HTML**")
        with col2:
            st.markdown("**🎨 CSS/SCSS**")
            st.markdown("**⚛️ JSX/TSX**")
            st.markdown("... y más en desarrollo")
    
    upload_method = st.radio(
        "Método de carga:",
        ["📦 Archivo ZIP", "📁 Directorio local"],
        horizontal=True
    )
    
    if upload_method == "📦 Archivo ZIP":
        uploaded_file = st.file_uploader(
            "Selecciona un archivo ZIP",
            type=['zip'],
            help="El ZIP se extraerá y guardará permanentemente"
        )
        
        if uploaded_file:
            # Mostrar info del archivo
            st.info(f"📄 **Archivo:** {uploaded_file.name}")
            st.info(f"💾 **Tamaño:** {len(uploaded_file.getvalue()) / 1024:.2f} KB")
            
            # Guardar ZIP temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            with st.spinner("📁 Procesando repositorio..."):
                try:
                    # Cargar desde ZIP
                    repo = st.session_state.repo_service.load_from_zip(tmp_path)
                    
                    if repo and repo.files:
                        st.success(f"✅ **Repositorio '{repo.name}' procesado correctamente**")
                        
                        # Estadísticas del repositorio
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📄 Archivos", len(repo.files))
                        with col2:
                            total_funcs = sum(len(f.functions) for f in repo.files)
                            st.metric("🔧 Funciones", total_funcs)
                        with col3:
                            total_classes = sum(len(f.classes) for f in repo.files)
                            st.metric("📚 Clases", total_classes)
                        with col4:
                            st.metric("📊 Líneas", repo.total_lines)
                        
                        # Distribución por lenguaje
                        languages = {}
                        for file in repo.files:
                            ext = file.extension
                            if ext not in languages:
                                languages[ext] = 0
                            languages[ext] += 1
                        
                        if languages:
                            st.subheader("📊 Distribución por lenguaje")
                            cols = st.columns(min(len(languages), 4))
                            for idx, (ext, count) in enumerate(sorted(languages.items(), key=lambda x: x[1], reverse=True)):
                                with cols[idx % 4]:
                                    st.metric(ext, count)
                        
                        # Indexar con Gemini
                        with st.spinner("🧠 Indexando con Gemini + FAISS..."):
                            try:
                                # Crear servicio RAG con caché integrado
                                rag_service = RAGService(
                                    repo_name=repo.name,
                                    repo_path=repo.path,
                                    repo_id=repo.db_id if hasattr(repo, 'db_id') else 0,
                                    prefer_pro=st.session_state.get('prefer_pro', False)
                                )
                                
                                if rag_service.index_repository(repo):
                                    st.success("✅ **Indexación completada**")
                                    st.session_state.rag_service = rag_service
                                    st.session_state.repository_loaded = True
                                    st.session_state.current_repo = repo
                                    
                                    # Mostrar estadísticas de indexación
                                    stats = rag_service.get_stats()
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("🎯 Fragmentos", stats['vector_store']['total_vectors'])
                                    with col2:
                                        st.metric("💾 Caché", f"{stats['cache']['total_files']} archivos")
                                    with col3:
                                        st.metric("🤖 Modelo", stats['llm']['current_model'][:20])
                                    
                                    # Mostrar uso de caché
                                    if stats['cache']['usage_percent'] > 0:
                                        st.progress(
                                            stats['cache']['usage_percent'] / 100,
                                            text=f"📦 Uso de caché: {stats['cache']['usage_percent']:.1f}%"
                                        )
                                else:
                                    st.error("❌ **Error en indexación**")
                                    st.info("Verifica los logs para más detalles")
                                    
                            except Exception as e:
                                st.error(f"❌ **Error en Gemini:** {str(e)}")
                                logger.error(f"Error de indexación: {e}", exc_info=True)
                    else:
                        st.error("❌ **No se encontraron archivos válidos**")
                        st.info("El ZIP debe contener archivos de código (.py, .js, .html, .css)")
                        
                except Exception as e:
                    st.error(f"❌ **Error:** {str(e)}")
                    with st.expander("🔍 Ver detalles del error"):
                        st.exception(e)
                finally:
                    os.unlink(tmp_path)
    
    else:  # Directorio local
        repo_path = st.text_input(
            "📁 Ruta del directorio:",
            placeholder="C:/Users/usuario/mi-repositorio",
            help="Ruta a un directorio local con código fuente"
        )
        
        if repo_path:
            path = Path(repo_path)
            if path.exists() and path.is_dir():
                with st.spinner("📁 Analizando repositorio..."):
                    repo = st.session_state.repo_service.load_from_directory(path)
                    
                    if repo and repo.files:
                        st.success(f"✅ **Repositorio '{repo.name}' procesado correctamente**")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📄 Archivos", len(repo.files))
                        with col2:
                            total_funcs = sum(len(f.functions) for f in repo.files)
                            st.metric("🔧 Funciones", total_funcs)
                        with col3:
                            total_classes = sum(len(f.classes) for f in repo.files)
                            st.metric("📚 Clases", total_classes)
                        with col4:
                            st.metric("📊 Líneas", repo.total_lines)
                        
                        with st.spinner("🧠 Indexando con Gemini..."):
                            try:
                                rag_service = RAGService(
                                    repo_name=repo.name,
                                    repo_path=path,
                                    repo_id=repo.db_id if hasattr(repo, 'db_id') else 0,
                                    prefer_pro=st.session_state.get('prefer_pro', False)
                                )
                                
                                if rag_service.index_repository(repo):
                                    st.success("✅ **Indexación completada**")
                                    st.session_state.rag_service = rag_service
                                    st.session_state.repository_loaded = True
                                    st.session_state.current_repo = repo
                                else:
                                    st.error("❌ **Error en indexación**")
                            except Exception as e:
                                st.error(f"❌ **Error:** {str(e)}")
                    else:
                        st.error("❌ **No se encontraron archivos válidos**")
            else:
                st.error("❌ **Directorio no válido**")
                st.info("Verifica que la ruta existe y es un directorio")


def show_chat_section() -> None:
    """Sección de chat con estadísticas de caché."""
    st.title("💬 Chat con el Repositorio")
    
    if 'current_repo' not in st.session_state or not st.session_state.current_repo:
        st.warning("⚠️ **Primero carga un repositorio**")
        st.info("👉 Ve a la pestaña **Cargar Repositorio** para comenzar")
        return
    
    if 'rag_service' not in st.session_state:
        st.warning("⚠️ **El repositorio no está indexado**")
        st.info("👉 Vuelve a cargar el repositorio para indexarlo")
        return
    
    # Barra lateral con estadísticas
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 Estadísticas del Repositorio")
        
        repo = st.session_state.current_repo
        st.info(f"📁 **{repo.name}**")
        st.write(f"📄 Archivos: {len(repo.files)}")
        
        # Estadísticas del RAG
        rag_stats = st.session_state.rag_service.get_stats()
        
        st.subheader("🎯 Vector Store")
        st.write(f"Fragmentos: {rag_stats['vector_store']['total_vectors']}")
        st.write(f"Dimensión: {rag_stats['embedding_dimension']}")
        
        st.subheader("💾 Caché")
        st.write(f"Archivos cacheados: {rag_stats['cache']['total_files']}")
        st.write(f"Uso: {rag_stats['cache']['usage_percent']:.1f}%")
        st.write(f"Total hits: {rag_stats['cache']['total_hits']}")
        
        # Barra de progreso de caché
        st.progress(
            rag_stats['cache']['usage_percent'] / 100,
            text=f"📦 Uso de caché"
        )
        
        st.subheader("🔍 Consultas")
        qc = rag_stats['query_cache']
        st.write(f"Cache hits: {qc['hits']}")
        st.write(f"Cache misses: {qc['misses']}")
        st.write(f"Hit rate: {qc['hit_rate']}%")
        
        st.subheader("🤖 Modelo")
        st.write(f"Actual: {rag_stats['llm']['current_model']}")
        st.write(f"Tipo: {'⭐ PRO' if rag_stats['llm']['model_type'] == 'pro' else '⚡ FLASH'}")
        
        # Opción para cambiar modelo
        current_pref = st.session_state.get('prefer_pro', False)
        new_pref = st.checkbox(
            "✨ **Preferir modelo Pro**",
            value=current_pref,
            help="Los modelos Pro son más capaces pero pueden tener rate limits"
        )
        
        if new_pref != current_pref:
            st.session_state['prefer_pro'] = new_pref
            st.rerun()
    
    # Historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Mostrar mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "model_used" in message and message["role"] == "assistant":
                st.caption(f"🤖 {message['model_used']}")
            if "cache_hit" in message and message["role"] == "assistant":
                if message["cache_hit"]:
                    st.caption("⚡ Respuesta desde caché")
            if "sources" in message and message["sources"]:
                with st.expander("📚 Fuentes consultadas"):
                    for source in message["sources"]:
                        st.write(f"**📄 {source['file']}** (score: {source.get('score', 'N/A')})")
                        st.code(source['preview'], language='python')
    
    # Input
    if prompt := st.chat_input("💬 Pregunta sobre el código..."):
        # Mostrar pregunta
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("🔍 Buscando en el código..."):
                try:
                    result = st.session_state.rag_service.query(
                        question=prompt,
                        k=st.session_state.get('k_results', 5),
                        include_sources=True,
                        use_cache=True
                    )
                    
                    answer = result['answer']
                    sources = result['sources']
                    model_used = result.get('model_used', 'desconocido')
                    elapsed = result.get('elapsed_seconds', 0)
                    
                    st.markdown(answer)
                    
                    # Mostrar métricas
                    cols = st.columns(3)
                    with cols[0]:
                        st.caption(f"🤖 {model_used}")
                    with cols[1]:
                        st.caption(f"⏱️ {elapsed:.2f} seg")
                    with cols[2]:
                        st.caption(f"📚 {len(sources)} fuentes")
                    
                    # Mostrar caché hit rate
                    if 'cache_stats' in result:
                        cache_stats = result['cache_stats']
                        st.caption(f"💾 Cache hit rate: {cache_stats['hit_rate']}%")
                    
                    if sources:
                        with st.expander("📚 Fuentes consultadas"):
                            for source in sources:
                                st.write(f"**📄 {source['file']}**")
                                st.code(source['preview'], language='python')
                    
                    # Guardar en historial
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "model_used": model_used,
                        "elapsed": elapsed,
                        "cache_hit": result.get('cache_stats', {}).get('hit_rate', 0) > 0
                    })
                    
                except Exception as e:
                    st.error(f"❌ **Error:** {str(e)}")
                    logger.error(f"Error en chat: {e}", exc_info=True)


def show_analysis_section() -> None:
    """Sección de análisis con vista por lenguaje."""
    st.title("📊 Análisis del Repositorio")
    
    if 'current_repo' not in st.session_state or not st.session_state.current_repo:
        st.warning("⚠️ **Primero carga un repositorio**")
        st.info("👉 Ve a la pestaña **Cargar Repositorio** para comenzar")
        return
    
    repo = st.session_state.current_repo
    
    # Métricas generales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Archivos", len(repo.files))
    with col2:
        total_funcs = sum(len(f.functions) for f in repo.files)
        st.metric("🔧 Funciones", total_funcs)
    with col3:
        total_classes = sum(len(f.classes) for f in repo.files)
        st.metric("📚 Clases", total_classes)
    with col4:
        st.metric("📊 Líneas", repo.total_lines)
    
    # Distribución por extensión
    extension_counts = {}
    lang_counts = {}
    
    for file in repo.files:
        ext = file.extension
        extension_counts[ext] = extension_counts.get(ext, 0) + 1
    
    if extension_counts:
        st.subheader("📊 Distribución por extensión")
        cols = st.columns(min(len(extension_counts), 5))
        for idx, (ext, count) in enumerate(sorted(extension_counts.items(), key=lambda x: x[1], reverse=True)):
            with cols[idx % 5]:
                st.metric(ext, count)
    
    # Lista de archivos con filtro por lenguaje
    st.subheader("📁 Archivos del repositorio")
    
    # Filtro por extensión
    extensions = sorted(set(f.extension for f in repo.files))
    selected_ext = st.selectbox("Filtrar por extensión:", ["Todos"] + extensions)
    
    # Filtrar archivos
    files_to_show = repo.files
    if selected_ext != "Todos":
        files_to_show = [f for f in repo.files if f.extension == selected_ext]
    
    # Mostrar archivos
    for file in files_to_show[:50]:  # Limitar a 50 para rendimiento
        with st.expander(f"📄 {file.name}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Líneas:** {file.line_count}")
            with col2:
                st.write(f"**Funciones:** {len(file.functions)}")
            with col3:
                st.write(f"**Clases:** {len(file.classes)}")
            
            if file.functions:
                st.write("**🔧 Funciones:**")
                for func in file.functions[:10]:
                    st.write(f"- `{func.name}` (líneas {func.line_start}-{func.line_end})")
            
            if file.classes:
                st.write("**📚 Clases:**")
                for cls in file.classes[:5]:
                    st.write(f"- `{cls.name}`")
                    if cls.methods:
                        for method in cls.methods[:3]:
                            st.write(f"  - método: `{method.name}()`")
    
    if len(files_to_show) > 50:
        st.info(f"📊 Mostrando 50 de {len(files_to_show)} archivos")


def show_repositories_list() -> None:
    """Muestra lista de repositorios con estadísticas de caché."""
    st.title("📚 Repositorios Analizados")
    
    if 'repo_service' not in st.session_state:
        st.session_state.repo_service = RepositoryService()
    
    repos = st.session_state.repo_service.list_repositories()
    
    if not repos:
        st.info("📭 **No hay repositorios analizados**")
        st.info("👉 Ve a la pestaña **Cargar Repositorio** para comenzar")
        return
    
    for repo in repos:
        with st.container():
            cols = st.columns([3, 1, 1, 1, 1, 1])
            
            with cols[0]:
                st.write(f"**{repo['name']}**")
                # Mostrar si los archivos existen
                if repo.get('files_exist', False):
                    st.caption(f"✅ Archivos en disco ({repo.get('size_mb', 0)} MB)")
                else:
                    st.caption("⚠️ Archivos no encontrados")
            
            with cols[1]:
                st.write(f"📁 {repo['file_count']}")
            with cols[2]:
                st.write(f"📊 {repo['total_lines']}")
            with cols[3]:
                # Formatear fecha
                created_at = repo['created_at']
                if hasattr(created_at, 'strftime'):
                    fecha = created_at.strftime("%Y-%m-%d")
                else:
                    fecha = str(created_at)[:10]
                st.write(f"🕐 {fecha}")
            with cols[4]:
                if st.button("📊 Estadísticas", key=f"stats_{repo['id']}"):
                    # Mostrar estadísticas en un popup
                    st.info("Estadísticas en desarrollo")
            with cols[5]:
                if st.button("🗑️ Eliminar", key=f"delete_{repo['id']}"):
                    if st.session_state.repo_service.delete_repository(repo['id']):
                        st.success(f"Repositorio {repo['name']} eliminado")
                        st.rerun()
                    else:
                        st.error("Error al eliminar")
            
            st.divider()


def show_configuration_section() -> None:
    """Configuración avanzada con estadísticas de caché."""
    st.title("⚙️ Configuración")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔑 API Key", "🤖 Modelos", "💾 Caché", "📁 Sistema"])
    
    with tab1:
        st.subheader("🔑 API Key de Gemini")
        
        current_key = os.getenv("GEMINI_API_KEY", "")
        masked = current_key[:10] + "..." + current_key[-5:] if current_key else "No configurada"
        
        st.info(f"API Key actual: `{masked}`")
        
        new_key = st.text_input(
            "Nueva API Key:",
            type="password",
            placeholder="AIzaSy...",
            help="Ingresa tu API key de Google AI Studio"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Guardar API Key", use_container_width=True):
                if new_key:
                    if save_env_var("GEMINI_API_KEY", new_key):
                        st.success("✅ API Key guardada correctamente")
                        st.rerun()
                    else:
                        st.error("❌ Error guardando API Key")
                else:
                    st.warning("Ingresa una API Key")
        
        with col2:
            if st.button("🔄 Probar conexión", use_container_width=True):
                try:
                    test_llm = GeminiLLM()
                    models = test_llm.list_available_models()
                    st.success(f"✅ Conexión exitosa!")
                    st.write(f"Modelos disponibles: {len(models)}")
                    for m in models[:3]:
                        st.write(f"- `{m}`")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        st.markdown("---")
        st.markdown("""
        **📝 Cómo obtener tu API Key:**
        1. Ve a [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
        2. Inicia sesión con tu cuenta Google
        3. Crea una nueva API key
        4. Cópiala y pégala aquí
        """)
    
    with tab2:
        st.subheader("🤖 Configuración de Modelos")
        
        prefer_pro = st.radio(
            "Preferencia de modelo:",
            ["⚡ Flash (gratuito, rápido)", "⭐ Pro (más capaz)"],
            index=0 if not st.session_state.get('prefer_pro', False) else 1,
            help="Flash usa gemini-2.0-flash, Pro usa gemini-2.5-pro"
        )
        
        st.session_state['prefer_pro'] = ("Pro" in prefer_pro)
        
        temperature = st.slider(
            "🌡️ Temperatura (creatividad):",
            0.0, 1.0, st.session_state.get('temperature', 0.2), 0.1,
            help="Valores bajos = más preciso, valores altos = más creativo"
        )
        st.session_state['temperature'] = temperature
        
        k_results = st.slider(
            "📚 Fragmentos a recuperar:",
            1, 20, st.session_state.get('k_results', 5),
            help="Número de fragmentos de código a usar como contexto"
        )
        st.session_state['k_results'] = k_results
        
        if st.button("💾 Guardar configuración", use_container_width=True):
            st.success("✅ Configuración guardada")
    
    with tab3:
        st.subheader("💾 Configuración de Caché")
        
        if 'repo_service' in st.session_state:
            cache_stats = st.session_state.repo_service.cache.get_stats()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📦 Archivos en caché", cache_stats['total_files'])
            with col2:
                st.metric("💾 Tamaño total", f"{cache_stats['total_size_mb']} MB")
            with col3:
                st.metric("📊 Uso", f"{cache_stats['usage_percent']}%")
            
            st.progress(
                cache_stats['usage_percent'] / 100,
                text=f"Uso de caché"
            )
            
            st.write("**⭐ Archivos más accedidos:**")
            for f in cache_stats.get('most_accessed', [])[:5]:
                st.write(f"- `{f['path']}` ({f['hits']} accesos, {f['size_kb']:.1f} KB)")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Limpiar todo el caché", use_container_width=True):
                    st.session_state.repo_service.cache.clear_all()
                    st.success("✅ Caché limpiado")
                    st.rerun()
            
            with col2:
                if st.button("📊 Ver estadísticas detalladas", use_container_width=True):
                    st.json(cache_stats)
    
    with tab4:
        st.subheader("📁 Información del Sistema")
        
        # Directorios
        st.write("**Directorios de datos:**")
        st.write(f"- 📁 Repositorios: `{Path('data/repositories').absolute()}`")
        st.write(f"- 💾 Vector store: `{Path('data/vectors').absolute()}`")
        st.write(f"- 🗄️ Caché: `{Path('data/cache').absolute()}`")
        
        # Variables de entorno
        st.write("**Configuración actual:**")
        st.write(f"- DB_HOST: {os.getenv('DB_HOST', 'localhost')}")
        st.write(f"- DB_NAME: {os.getenv('DB_NAME', 'ai_coding_assistant')}")
        st.write(f"- Gemini API: {'✅ Configurada' if os.getenv('GEMINI_API_KEY') else '❌ No configurada'}")
        
        # Estadísticas del servicio
        if 'repo_service' in st.session_state:
            stats = st.session_state.repo_service.get_repository_stats()
            
            st.write("**Estadísticas del servicio:**")
            st.write(f"- Repositorios totales: {stats['repositories']['total_repos']}")
            st.write(f"- Archivos totales: {stats['repositories']['total_files']}")
            st.write(f"- Funciones totales: {stats['repositories']['total_functions']}")
            st.write(f"- Clases totales: {stats['repositories']['total_classes']}")
            
            if stats['repositories']['languages']:
                st.write("**Lenguajes procesados:**")
                for lang, count in stats['repositories']['languages'].items():
                    st.write(f"- {lang}: {count} archivos")
        
        # Botón de limpieza
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Limpiar vectores", use_container_width=True):
                import shutil
                vectors_dir = Path("data/vectors")
                if vectors_dir.exists():
                    shutil.rmtree(vectors_dir)
                    vectors_dir.mkdir()
                    st.success("✅ Vectores limpiados")
                    st.rerun()
        
        with col2:
            if st.button("📝 Ver logs", use_container_width=True):
                log_file = Path("ai_coding_assistant.log")
                if log_file.exists():
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()[-50:]
                        st.code(''.join(lines))
                else:
                    st.warning("No hay logs")


def main() -> None:
    """Función principal de la interfaz."""
    
    # Inicializar estado de sesión
    if 'prefer_pro' not in st.session_state:
        st.session_state.prefer_pro = False
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.2
    if 'k_results' not in st.session_state:
        st.session_state.k_results = 5
    
    # Barra lateral de navegación
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/robot.png", width=80)
        st.title("🤖 AI Coding Assistant")
        
        st.markdown("---")
        
        # Navegación
        page = st.radio(
            "📋 **Navegación**",
            [
                "📤 Cargar Repositorio",
                "📊 Analizar",
                "💬 Chat",
                "📚 Repositorios",
                "⚙️ Configuración"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Estado del sistema
        st.subheader("📊 Estado")
        if 'current_repo' in st.session_state and st.session_state.current_repo:
            st.success(f"✅ **Activo:** {st.session_state.current_repo.name}")
            
            # Métricas rápidas
            repo = st.session_state.current_repo
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Archivos", len(repo.files))
            with col2:
                st.metric("📊 Líneas", repo.total_lines)
        else:
            st.warning("⏳ **Sin repositorio activo**")
        
        st.markdown("---")
        st.caption("v1.0.0 | Gemini + FAISS")
        st.caption("Multi-lenguaje | Caché LRU")
    
    # Mostrar página seleccionada
    if page == "📤 Cargar Repositorio":
        show_upload_section()
    elif page == "📊 Analizar":
        show_analysis_section()
    elif page == "💬 Chat":
        show_chat_section()
    elif page == "📚 Repositorios":
        show_repositories_list()
    else:  # Configuración
        show_configuration_section()


if __name__ == "__main__":
    main()