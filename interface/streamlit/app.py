"""
Módulo de interfaz de usuario para AI Coding Assistant Local.
Versión final con manejo de repositorios como diccionario u objeto.
"""

import streamlit as st
import logging
from pathlib import Path
import tempfile
import os
from dotenv import load_dotenv, set_key
from datetime import datetime
from typing import Union, Dict, Any, List

from application.services.repo_service import RepositoryService
from application.services.rag_gemini_service import RAGService
from infrastructure.llm_clients.gemini_llm import GeminiLLM

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


def get_repo_files(repo: Union[Dict[str, Any], Any]) -> List:
    """
    Obtiene la lista de archivos del repositorio de manera segura.
    
    Args:
        repo: Repositorio (objeto o diccionario)
        
    Returns:
        Lista de archivos
    """
    if isinstance(repo, dict):
        return repo.get('files', [])
    else:
        return getattr(repo, 'files', [])


def get_repo_file_count(repo: Union[Dict[str, Any], Any]) -> int:
    """
    Obtiene el número de archivos del repositorio de manera segura.
    
    Args:
        repo: Repositorio (objeto o diccionario)
        
    Returns:
        Número de archivos
    """
    if isinstance(repo, dict):
        return repo.get('file_count', 0)
    else:
        return len(getattr(repo, 'files', [])) if hasattr(repo, 'files') else getattr(repo, 'file_count', 0)


def get_repo_total_lines(repo: Union[Dict[str, Any], Any]) -> int:
    """
    Obtiene el total de líneas del repositorio de manera segura.
    
    Args:
        repo: Repositorio (objeto o diccionario)
        
    Returns:
        Total de líneas
    """
    if isinstance(repo, dict):
        return repo.get('total_lines', 0)
    else:
        return getattr(repo, 'total_lines', 0)


def get_repo_name(repo: Union[Dict[str, Any], Any]) -> str:
    """
    Obtiene el nombre del repositorio de manera segura.
    
    Args:
        repo: Repositorio (objeto o diccionario)
        
    Returns:
        Nombre del repositorio
    """
    if isinstance(repo, dict):
        return repo.get('name', 'desconocido')
    else:
        return getattr(repo, 'name', 'desconocido')


def show_upload_section() -> None:
    """Sección de carga con filtros de archivos y estadísticas."""
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
    
    # Mostrar información de optimización
    with st.expander("⚡ Optimizaciones activas", expanded=False):
        st.markdown("""
        **Para respetar los límites de API gratuita:**
        - 📄 Máximo **10 fragmentos por archivo**
        - 📏 Fragmentos de **500 caracteres**
        - 💾 Archivos mayores a **1 MB** son ignorados
        - 📊 Archivos con más de **2000 líneas** son ignorados
        - ⏱️ Procesamiento en lotes de **20 fragmentos**
        - 🔄 Reintento automático en caso de límites de API
        """)
    
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
            file_size = len(uploaded_file.getvalue()) / 1024
            st.info(f"📄 **Archivo:** {uploaded_file.name}")
            st.info(f"💾 **Tamaño:** {file_size:.2f} KB")
            
            # Guardar ZIP temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            with st.spinner("📁 Procesando repositorio..."):
                try:
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
                        
                        # Mostrar distribución por lenguaje
                        languages = {}
                        for file in repo.files:
                            ext = file.extension
                            if ext not in languages:
                                languages[ext] = 0
                            languages[ext] += 1
                        
                        if languages:
                            st.subheader("📊 Distribución por extensión")
                            cols = st.columns(min(len(languages), 4))
                            for idx, (ext, count) in enumerate(sorted(languages.items(), key=lambda x: x[1], reverse=True)[:4]):
                                with cols[idx % 4]:
                                    st.metric(ext, count)
                        
                        # Indexar con Gemini
                        with st.spinner("🧠 Indexando con Gemini + FAISS (puede tomar unos minutos)..."):
                            try:
                                rag_service = RAGService(
                                    repo_name=repo.name,
                                    repo_path=repo.path,
                                    repo_id=repo.db_id if hasattr(repo, 'db_id') else 0,
                                    prefer_pro=st.session_state.get('prefer_pro', False),
                                    max_file_size_mb=st.session_state.get('max_file_size_mb', 1),
                                    include_docs=st.session_state.get('include_docs', False)
                                )
                                
                                if rag_service.index_repository(repo):
                                    st.success("✅ **Indexación completada**")
                                    st.session_state.rag_service = rag_service
                                    st.session_state.repository_loaded = True
                                    st.session_state.current_repo = repo
                                    
                                    # Limpiar historial de chat al cargar nuevo repositorio
                                    st.session_state.messages = []
                                    
                                    # Mostrar estadísticas de indexación
                                    stats = rag_service.get_stats()
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("🎯 Fragmentos", stats['processing_stats']['chunks_processed'])
                                    with col2:
                                        st.metric("📞 API Calls", stats['processing_stats']['api_calls'])
                                    with col3:
                                        st.metric("⚠️ Rate Limits", stats['processing_stats']['rate_limit_hits'])
                                    with col4:
                                        st.metric("📐 Dimensión", stats['embedding_dimension'])
                                    
                                    # Barra de progreso de fragmentos
                                    if stats['processing_stats']['total_chunks'] > 0:
                                        progress = stats['processing_stats']['chunks_processed'] / stats['processing_stats']['total_chunks']
                                        st.progress(progress, text=f"Procesamiento: {progress*100:.0f}%")
                                    
                                else:
                                    st.error("❌ **Error en indexación**")
                                    st.info("Verifica los logs para más detalles")
                                    
                            except Exception as e:
                                st.error(f"❌ **Error en Gemini:** {str(e)}")
                                logger.error(f"Error de indexación: {e}", exc_info=True)
                    else:
                        st.error("❌ **No se encontraron archivos válidos**")
                        st.info("El ZIP debe contener archivos de código con extensiones soportadas")
                        
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
                        
                        with st.spinner("🧠 Indexando con Gemini (puede tomar unos minutos)..."):
                            try:
                                rag_service = RAGService(
                                    repo_name=repo.name,
                                    repo_path=path,
                                    repo_id=repo.db_id if hasattr(repo, 'db_id') else 0,
                                    prefer_pro=st.session_state.get('prefer_pro', False),
                                    max_file_size_mb=st.session_state.get('max_file_size_mb', 1),
                                    include_docs=st.session_state.get('include_docs', False)
                                )
                                
                                if rag_service.index_repository(repo):
                                    st.success("✅ **Indexación completada**")
                                    st.session_state.rag_service = rag_service
                                    st.session_state.repository_loaded = True
                                    st.session_state.current_repo = repo
                                    
                                    # Limpiar historial de chat
                                    st.session_state.messages = []
                                    
                                    stats = rag_service.get_stats()
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("🎯 Fragmentos", stats['processing_stats']['chunks_processed'])
                                    with col2:
                                        st.metric("📞 API Calls", stats['processing_stats']['api_calls'])
                                    with col3:
                                        st.metric("⚠️ Rate Limits", stats['processing_stats']['rate_limit_hits'])
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
    """Sección de chat con estadísticas de procesamiento."""
    st.title("💬 Chat con el Repositorio")
    
    if 'current_repo' not in st.session_state or not st.session_state.current_repo:
        st.warning("⚠️ **Primero carga un repositorio**")
        st.info("👉 Ve a la pestaña **Cargar Repositorio** para comenzar")
        return
    
    if 'rag_service' not in st.session_state or not st.session_state.rag_service:
        st.warning("⚠️ **El repositorio no está indexado**")
        st.info("👉 Vuelve a cargar el repositorio para indexarlo")
        return
    
    # Barra lateral con estadísticas detalladas
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 Estadísticas del Repositorio")
        
        repo_name = get_repo_name(st.session_state.current_repo)
        repo_files = get_repo_file_count(st.session_state.current_repo)
        repo_lines = get_repo_total_lines(st.session_state.current_repo)
        
        st.info(f"📁 **{repo_name}**")
        st.write(f"📄 Archivos: {repo_files}")
        st.write(f"📊 Líneas: {repo_lines}")
        
        try:
            stats = st.session_state.rag_service.get_stats()
            
            st.subheader("🎯 Procesamiento")
            st.write(f"Fragmentos: {stats['processing_stats']['chunks_processed']}")
            st.write(f"Solicitudes API: {stats['processing_stats']['api_calls']}")
            st.write(f"Rate limit hits: {stats['processing_stats']['rate_limit_hits']}")
            
            if stats['processing_stats']['api_calls'] > 0:
                success_rate = (stats['processing_stats']['api_calls'] - stats['processing_stats']['rate_limit_hits']) / stats['processing_stats']['api_calls'] * 100
                st.write(f"Tasa éxito: {success_rate:.1f}%")
            
            st.subheader("📁 Límites activos")
            st.write(f"Fragmentos por archivo: {stats['limits']['max_fragments_per_file']}")
            st.write(f"Tamaño fragmento: {stats['limits']['max_chunk_size']} chars")
            st.write(f"Tamaño máximo archivo: {stats['limits']['max_file_size_mb']} MB")
            st.write(f"Dimensión embeddings: {stats['embedding_dimension']}")
            
            st.subheader("🤖 Modelo")
            model_type = stats['llm']['model_type']
            model_emoji = "⭐ PRO" if model_type == 'pro' else "⚡ FLASH"
            st.write(f"{model_emoji} {stats['llm']['current_model']}")
        except Exception as e:
            st.warning(f"No se pudieron obtener estadísticas: {e}")
        
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
            if "sources" in message and message["sources"]:
                with st.expander("📚 Fuentes consultadas"):
                    for source in message["sources"]:
                        st.write(f"**📄 {source['file']}** (score: {source.get('score', 'N/A')})")
                        st.code(source['preview'], language='python')
    
    # Input
    if prompt := st.chat_input("💬 Pregunta sobre el código..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Buscando en el código..."):
                try:
                    result = st.session_state.rag_service.query(
                        question=prompt,
                        k=st.session_state.get('k_results', 5),
                        include_sources=True
                    )
                    
                    answer = result['answer']
                    sources = result['sources']
                    model_used = result.get('model_used', 'desconocido')
                    elapsed = result.get('elapsed_seconds', 0)
                    
                    st.markdown(answer)
                    
                    cols = st.columns(3)
                    with cols[0]:
                        st.caption(f"🤖 {model_used}")
                    with cols[1]:
                        st.caption(f"⏱️ {elapsed:.2f} seg")
                    with cols[2]:
                        st.caption(f"📚 {len(sources)} fuentes")
                    
                    if sources:
                        with st.expander("📚 Fuentes consultadas"):
                            for source in sources:
                                st.write(f"**📄 {source['file']}**")
                                st.code(source['preview'], language='python')
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "model_used": model_used
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
    
    # Verificar si es diccionario o objeto
    if isinstance(repo, dict):
        # Es un diccionario (cargado desde BD)
        st.warning("⚠️ **Vista limitada - El repositorio fue cargado desde BD**")
        st.info("Para análisis completo, indexa el repositorio nuevamente desde 'Cargar Repositorio'")
        
        # Mostrar información básica del diccionario
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Archivos", repo.get('file_count', 0))
        with col2:
            st.metric("🔧 Funciones", repo.get('total_functions', 0) if 'total_functions' in repo else 0)
        with col3:
            st.metric("📚 Clases", repo.get('total_classes', 0) if 'total_classes' in repo else 0)
        with col4:
            st.metric("📊 Líneas", repo.get('total_lines', 0))
        
        st.info("💡 **Sugerencia:** Indexa el repositorio nuevamente desde 'Cargar Repositorio' para ver el análisis completo de archivos, funciones y clases.")
        return
    
    # Es un objeto (cargado desde indexación)
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
    for file in repo.files:
        ext = file.extension
        extension_counts[ext] = extension_counts.get(ext, 0) + 1
    
    if extension_counts:
        st.subheader("📊 Distribución por extensión")
        cols = st.columns(min(len(extension_counts), 5))
        for idx, (ext, count) in enumerate(sorted(extension_counts.items(), key=lambda x: x[1], reverse=True)):
            with cols[idx % 5]:
                st.metric(ext, count)
    
    # Lista de archivos con filtro
    st.subheader("📁 Archivos del repositorio")
    
    extensions = sorted(set(f.extension for f in repo.files))
    selected_ext = st.selectbox("Filtrar por extensión:", ["Todos"] + extensions)
    
    files_to_show = repo.files
    if selected_ext != "Todos":
        files_to_show = [f for f in repo.files if f.extension == selected_ext]
    
    for file in files_to_show[:50]:
        with st.expander(f"📄 {file.name}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Líneas:** {file.line_count}")
            with col2:
                st.write(f"**Funciones:** {len(file.functions)}")
            with col3:
                st.write(f"**Clases:** {len(file.classes)}")
            
            # Mostrar funciones
            if file.functions:
                st.write("**🔧 Funciones:**")
                for func in file.functions[:10]:
                    # Verificar si es diccionario o objeto
                    if isinstance(func, dict):
                        func_name = func.get('name', 'unknown')
                        line_start = func.get('line_start', 0)
                        line_end = func.get('line_end', 0)
                    else:
                        func_name = func.name
                        line_start = func.line_start
                        line_end = func.line_end
                    st.write(f"- `{func_name}` (líneas {line_start}-{line_end})")
            
            # Mostrar clases
            if file.classes:
                st.write("**📚 Clases:**")
                for cls in file.classes[:5]:
                    # Verificar si es diccionario o objeto
                    if isinstance(cls, dict):
                        cls_name = cls.get('name', 'unknown')
                        methods = cls.get('methods', [])
                    else:
                        cls_name = cls.name
                        methods = cls.methods
                    
                    st.write(f"- `{cls_name}`")
                    if methods:
                        for method in methods[:3]:
                            if isinstance(method, dict):
                                method_name = method.get('name', 'unknown')
                            else:
                                method_name = method.name
                            st.write(f"  - método: `{method_name}()`")
    
    if len(files_to_show) > 50:
        st.info(f"📊 Mostrando 50 de {len(files_to_show)} archivos")


def show_repositories_list() -> None:
    """Muestra lista de repositorios."""
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
            cols = st.columns([3, 1, 1, 1, 1])
            
            with cols[0]:
                st.write(f"**{repo['name']}**")
                path = Path(repo['path'])
                if path.exists():
                    st.caption(f"✅ Archivos en disco")
                else:
                    st.caption("⚠️ Archivos no encontrados")
            
            with cols[1]:
                st.write(f"📁 {repo['file_count']}")
            with cols[2]:
                st.write(f"📊 {repo['total_lines']}")
            with cols[3]:
                created_at = repo['created_at']
                if hasattr(created_at, 'strftime'):
                    fecha = created_at.strftime("%Y-%m-%d")
                else:
                    fecha = str(created_at)[:10]
                st.write(f"🕐 {fecha}")
            with cols[4]:
                if st.button("🗑️", key=f"delete_{repo['id']}"):
                    if st.session_state.repo_service.delete_repository(repo['id']):
                        st.success(f"Repositorio {repo['name']} eliminado")
                        st.rerun()
            
            st.divider()


def show_configuration_section() -> None:
    """Configuración avanzada con límites ajustables."""
    st.title("⚙️ Configuración")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔑 API Key", "🤖 Modelos", "📁 Límites", "📊 Sistema"])
    
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
                        st.success("✅ API Key guardada")
                        st.rerun()
                    else:
                        st.error("❌ Error guardando API Key")
                else:
                    st.warning("Ingresa una API Key")
        
        with col2:
            if st.button("🔄 Probar conexión", use_container_width=True):
                try:
                    test_llm = GeminiLLM()
                    st.success("✅ Conexión exitosa!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        st.markdown("---")
        st.markdown("""
        **📝 Límites de API gratuita:**
        - 100 solicitudes de embeddings por minuto
        - 60 solicitudes de texto por minuto
        - 1,500,000 tokens por día
        """)
    
    with tab2:
        st.subheader("🤖 Configuración de Modelos")
        
        prefer_pro = st.radio(
            "Preferencia de modelo:",
            ["⚡ Flash (gratuito, rápido)", "⭐ Pro (más capaz)"],
            index=0 if not st.session_state.get('prefer_pro', False) else 1
        )
        st.session_state['prefer_pro'] = ("Pro" in prefer_pro)
        
        temperature = st.slider(
            "🌡️ Temperatura:",
            0.0, 1.0, st.session_state.get('temperature', 0.2), 0.1,
            help="Valores bajos = más preciso, valores altos = más creativo"
        )
        st.session_state['temperature'] = temperature
        
        k_results = st.slider(
            "📚 Fragmentos a recuperar:",
            1, 10, st.session_state.get('k_results', 5),
            help="Número de fragmentos de código a usar como contexto"
        )
        st.session_state['k_results'] = k_results
    
    with tab3:
        st.subheader("📁 Límites de Procesamiento")
        
        st.markdown("**Límites actuales (optimizados para free tier):**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fragmentos por archivo", "10")
            st.metric("Tamaño fragmento", "500 caracteres")
            st.metric("Archivos por lote", "20 fragmentos")
        with col2:
            st.metric("Tamaño máximo archivo", "1 MB")
            st.metric("Líneas máximas", "2000 líneas")
            st.metric("Delay entre lotes", "2 segundos")
        
        st.markdown("---")
        
        max_file_size = st.slider(
            "📦 Tamaño máximo de archivo (MB):",
            min_value=1,
            max_value=10,
            value=st.session_state.get('max_file_size_mb', 1),
            help="Archivos más grandes serán ignorados"
        )
        st.session_state['max_file_size_mb'] = max_file_size
        
        include_docs = st.checkbox(
            "📚 Incluir archivos de documentación (.md, .rst)",
            value=st.session_state.get('include_docs', False),
            help="Aumenta el número de fragmentos procesados"
        )
        st.session_state['include_docs'] = include_docs
        
        if st.button("💾 Guardar configuración", use_container_width=True):
            st.success("✅ Configuración guardada")
    
    with tab4:
        st.subheader("📊 Información del Sistema")
        
        st.write("**Directorios de datos:**")
        st.write(f"- 📁 Repositorios: `{Path('data/repositories').absolute()}`")
        st.write(f"- 💾 Vector store: `{Path('data/vectors').absolute()}`")
        st.write(f"- 🗄️ Caché: `{Path('data/cache').absolute()}`")
        
        if 'repo_service' in st.session_state:
            try:
                stats = st.session_state.repo_service.get_repository_stats()
                st.write("**Estadísticas generales:**")
                st.write(f"- Repositorios totales: {stats['repositories']['total_repos']}")
                st.write(f"- Archivos totales: {stats['repositories']['total_files']}")
                st.write(f"- Funciones totales: {stats['repositories']['total_functions']}")
            except Exception as e:
                st.write(f"No se pudieron obtener estadísticas: {e}")
        
        if st.button("🧹 Limpiar vectores", use_container_width=True):
            import shutil
            vectors_dir = Path("data/vectors")
            if vectors_dir.exists():
                shutil.rmtree(vectors_dir)
                vectors_dir.mkdir()
                st.success("✅ Vectores limpiados")
                st.rerun()


def main() -> None:
    """Función principal de la interfaz."""
    
    # Inicializar estado de sesión
    if 'prefer_pro' not in st.session_state:
        st.session_state.prefer_pro = False
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.2
    if 'k_results' not in st.session_state:
        st.session_state.k_results = 5
    if 'max_file_size_mb' not in st.session_state:
        st.session_state.max_file_size_mb = 1
    if 'include_docs' not in st.session_state:
        st.session_state.include_docs = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Barra lateral de navegación
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/robot.png", width=80)
        st.title("🤖 AI Coding Assistant")
        
        st.markdown("---")
        
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
            repo_name = get_repo_name(st.session_state.current_repo)
            repo_files = get_repo_file_count(st.session_state.current_repo)
            repo_lines = get_repo_total_lines(st.session_state.current_repo)
            
            st.success(f"✅ **Activo:** {repo_name}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Archivos", repo_files)
            with col2:
                st.metric("📊 Líneas", repo_lines)
            
            # Mostrar estadísticas de procesamiento si RAG existe
            if 'rag_service' in st.session_state and st.session_state.rag_service:
                try:
                    stats = st.session_state.rag_service.get_stats()
                    st.metric("🎯 Fragmentos", stats['processing_stats']['chunks_processed'])
                except Exception:
                    pass
        else:
            st.warning("⏳ **Sin repositorio activo**")
        
        st.markdown("---")
        st.caption("v1.0.0 | Gemini + FAISS")
        st.caption("Optimizado para free tier | Dimensión 3072")
    
    # Mostrar página seleccionada
    if page == "📤 Cargar Repositorio":
        show_upload_section()
    elif page == "📊 Analizar":
        show_analysis_section()
    elif page == "💬 Chat":
        show_chat_section()
    elif page == "📚 Repositorios":
        show_repositories_list()
    else:
        show_configuration_section()


if __name__ == "__main__":
    main()