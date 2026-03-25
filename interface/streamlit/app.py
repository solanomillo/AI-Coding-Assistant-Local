"""
Módulo de interfaz de usuario para AI Coding Assistant Local.
Contiene solo las funciones de UI. El sidebar y navegación están en main.py.
"""

import streamlit as st
import logging
from pathlib import Path
import tempfile
import os
from dotenv import load_dotenv, set_key
from typing import Union, Dict, Any, List

from application.services.repo_service import RepositoryService
from application.services.service_factory import ServiceFactory

logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()
ENV_FILE = ".env"


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
    """Obtiene la lista de archivos del repositorio."""
    if isinstance(repo, dict):
        return repo.get('files', [])
    else:
        return getattr(repo, 'files', [])


def get_repo_file_count(repo: Union[Dict[str, Any], Any]) -> int:
    """Obtiene el número de archivos."""
    if isinstance(repo, dict):
        return repo.get('file_count', 0)
    else:
        return len(getattr(repo, 'files', [])) if hasattr(repo, 'files') else getattr(repo, 'file_count', 0)


def get_repo_total_lines(repo: Union[Dict[str, Any], Any]) -> int:
    """Obtiene el total de líneas."""
    if isinstance(repo, dict):
        return repo.get('total_lines', 0)
    else:
        return getattr(repo, 'total_lines', 0)


def get_repo_name(repo: Union[Dict[str, Any], Any]) -> str:
    """Obtiene el nombre del repositorio."""
    if isinstance(repo, dict):
        return repo.get('name', 'desconocido')
    else:
        return getattr(repo, 'name', 'desconocido')


def _setup_services(repo) -> tuple:
    """
    Configura los servicios para un repositorio.
    Retorna (success, rag_service, agent_workflow)
    """
    if not ServiceFactory.is_api_key_valid():
        return False, None, None
    
    rag_service, agent_workflow = ServiceFactory.setup_repository_services(
        repo=repo,
        prefer_pro=st.session_state.get('prefer_pro', False),
        max_file_size_mb=st.session_state.get('max_file_size_mb', 1),
        include_docs=st.session_state.get('include_docs', False)
    )
    
    if rag_service is None or agent_workflow is None:
        return False, None, None
    
    return True, rag_service, agent_workflow


def _show_repository_stats(repo) -> None:
    """Muestra estadísticas del repositorio."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Archivos", len(repo.files))
    with col2:
        total_funcs = sum(len(f.functions) for f in repo.files)
        st.metric("🔧 Funciones", total_funcs)
    with col3:
        total_classes = sum(len(f.classes) for f in repo.files)
        st.metric("📚 Clases", total_classes)


def _check_quota_and_display() -> bool:
    """
    Verifica si el límite de API ha sido alcanzado y muestra mensaje.
    Retorna True si el límite fue alcanzado.
    """
    if st.session_state.get('daily_limit_reached', False):
        st.error("⚠️ **Límite diario de API alcanzado**")
        st.info("""
        Has superado el límite diario de solicitudes de Gemini API.
        
        **¿Qué puedes hacer?**
        - Esperar hasta mañana para que se reinicie el contador
        - Seguir usando repositorios ya indexados para consultas
        - Las nuevas indexaciones no funcionarán hasta mañana
        
        **Límites del free tier:**
        - 20 solicitudes de texto por minuto
        - 100 solicitudes de embeddings por minuto
        - 1500 solicitudes por día (aproximadamente)
        """)
        return True
    return False


def show_upload_section() -> None:
    """Sección de carga con verificación de duplicados."""
    st.title("📤 Cargar Repositorio")
    
    if 'repo_service' not in st.session_state:
        st.session_state.repo_service = RepositoryService()
    
    # Verificar API key al inicio
    if not ServiceFactory.is_api_key_valid():
        st.error("🔑 **API Key no configurada**")
        st.info("""
        Para usar esta aplicación, necesitas configurar tu API key de Gemini.
        
        **Pasos:**
        1. Ve a la pestaña **Configuración** en el menú lateral
        2. Ingresa tu API key de Google AI Studio
        3. Haz clic en Guardar
        
        **¿No tienes API key?** 
        - Ve a [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
        - Inicia sesión con tu cuenta Google
        - Crea una nueva API key (es gratuita)
        """)
        return
    
    # Verificar límite diario
    _check_quota_and_display()
    
    with st.expander("⚡ Optimizaciones activas", expanded=False):
        st.markdown("""
        **Para respetar los límites de API gratuita:**
        - 📄 Máximo **10 fragmentos por archivo**
        - 📏 Fragmentos de **500 caracteres**
        - 💾 Archivos mayores a **1 MB** son ignorados
        - 📊 Archivos con más de **2000 líneas** son ignorados
        - ⏱️ Procesamiento en lotes de **20 fragmentos**
        - 🤖 **Agentes especializados** con LangGraph
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
            file_size = len(uploaded_file.getvalue()) / 1024
            st.info(f"📄 **Archivo:** {uploaded_file.name}")
            st.info(f"💾 **Tamaño:** {file_size:.2f} KB")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            with st.spinner("📁 Procesando repositorio..."):
                try:
                    repo = st.session_state.repo_service.load_from_zip(tmp_path)
                    
                    if repo:
                        if hasattr(repo, 'db_id') and repo.db_id:
                            # Verificar si ya tiene vectores
                            safe_name = repo.name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                            index_path = Path(f"data/vectors/{safe_name}.index")
                            
                            if index_path.exists():
                                # Repositorio ya indexado
                                st.warning("⚠️ **Repositorio ya existente**")
                                st.info(f"El repositorio '{repo.name}' ya estaba indexado. Se ha cargado desde la base de datos.")
                                st.info(f"📁 Ruta: {repo.path}")
                                
                                _show_repository_stats(repo)
                                
                                success, rag_service, agent_workflow = _setup_services(repo)
                                
                                if success:
                                    st.session_state.rag_service = rag_service
                                    st.session_state.agent_workflow = agent_workflow
                                    st.session_state.current_repo = repo
                                    st.session_state.repository_loaded = True
                                    st.session_state.messages = []
                                    st.success("✅ Repositorio cargado correctamente desde caché")
                                else:
                                    st.error("❌ **Error al configurar los servicios**")
                                    st.info("Verifica que tu API key sea válida y que tengas conexión a internet.")
                            else:
                                # Repositorio recién guardado en BD pero sin vectores - hay que indexar
                                st.success(f"✅ **Repositorio '{repo.name}' procesado correctamente**")
                                
                                _show_repository_stats(repo)
                                
                                with st.spinner("🧠 Indexando con Gemini + FAISS..."):
                                    try:
                                        rag_service = ServiceFactory.create_rag_service(
                                            repo_name=repo.name,
                                            repo_path=repo.path,
                                            repo_id=repo.db_id,
                                            prefer_pro=st.session_state.get('prefer_pro', False),
                                            max_file_size_mb=st.session_state.get('max_file_size_mb', 1),
                                            include_docs=st.session_state.get('include_docs', False)
                                        )
                                        
                                        if rag_service is None:
                                            st.error("❌ **Error al crear el servicio RAG**")
                                            st.info("Verifica que tu API key sea válida y que tengas conexión a internet.")
                                            return
                                        
                                        if rag_service.index_repository(repo):
                                            st.success("✅ **Indexación completada**")
                                            
                                            success, rag_service, agent_workflow = _setup_services(repo)
                                            
                                            if success:
                                                st.session_state.rag_service = rag_service
                                                st.session_state.agent_workflow = agent_workflow
                                                st.session_state.current_repo = repo
                                                st.session_state.repository_loaded = True
                                                st.session_state.messages = []
                                                st.success("🤖 **Agentes LangGraph inicializados**")
                                            else:
                                                st.warning("⚠️ **Agentes no inicializados**")
                                                st.info("Los agentes requieren API key válida. Configúrala en Configuración.")
                                        else:
                                            st.error("❌ **Error en indexación**")
                                            
                                    except Exception as e:
                                        error_msg = str(e).lower()
                                        if 'quota' in error_msg or '429' in error_msg or 'rate limit' in error_msg:
                                            st.error("⚠️ **Límite de API alcanzado**")
                                            st.info("Has superado el límite diario de solicitudes. Las consultas estarán disponibles mañana.")
                                            st.session_state.daily_limit_reached = True
                                        else:
                                            st.error(f"❌ **Error en Gemini:** {str(e)}")
                                            st.info("Verifica tu API key en la pestaña Configuración.")
                                        logger.error(f"Error de indexación: {e}", exc_info=True)
                        else:
                            st.error("❌ **Error: Repositorio sin ID**")
                    else:
                        st.error("❌ **No se encontraron archivos válidos**")
                        
                except Exception as e:
                    st.error(f"❌ **Error:** {str(e)}")
                    with st.expander("🔍 Ver detalles"):
                        st.exception(e)
                finally:
                    os.unlink(tmp_path)
    
    else:
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
                    
                    if repo:
                        if hasattr(repo, 'db_id') and repo.db_id:
                            # Verificar si ya tiene vectores
                            safe_name = repo.name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                            index_path = Path(f"data/vectors/{safe_name}.index")
                            
                            if index_path.exists():
                                # Repositorio ya indexado
                                st.warning("⚠️ **Repositorio ya existente**")
                                st.info(f"El repositorio '{repo.name}' ya estaba indexado. Se ha cargado desde la base de datos.")
                                st.info(f"📁 Ruta: {repo.path}")
                                
                                _show_repository_stats(repo)
                                
                                success, rag_service, agent_workflow = _setup_services(repo)
                                
                                if success:
                                    st.session_state.rag_service = rag_service
                                    st.session_state.agent_workflow = agent_workflow
                                    st.session_state.current_repo = repo
                                    st.session_state.repository_loaded = True
                                    st.session_state.messages = []
                                    st.success("✅ Repositorio cargado correctamente desde caché")
                                else:
                                    st.error("❌ **Error al configurar los servicios**")
                                    st.info("Verifica que tu API key sea válida y que tengas conexión a internet.")
                            else:
                                # Repositorio recién guardado sin vectores
                                st.success(f"✅ **Repositorio '{repo.name}' procesado correctamente**")
                                
                                _show_repository_stats(repo)
                                
                                with st.spinner("🧠 Indexando con Gemini..."):
                                    try:
                                        rag_service = ServiceFactory.create_rag_service(
                                            repo_name=repo.name,
                                            repo_path=path,
                                            repo_id=repo.db_id,
                                            prefer_pro=st.session_state.get('prefer_pro', False),
                                            max_file_size_mb=st.session_state.get('max_file_size_mb', 1),
                                            include_docs=st.session_state.get('include_docs', False)
                                        )
                                        
                                        if rag_service is None:
                                            st.error("❌ **Error al crear el servicio RAG**")
                                            st.info("Verifica que tu API key sea válida y que tengas conexión a internet.")
                                            return
                                        
                                        if rag_service.index_repository(repo):
                                            st.success("✅ **Indexación completada**")
                                            
                                            success, rag_service, agent_workflow = _setup_services(repo)
                                            
                                            if success:
                                                st.session_state.rag_service = rag_service
                                                st.session_state.agent_workflow = agent_workflow
                                                st.session_state.current_repo = repo
                                                st.session_state.repository_loaded = True
                                                st.session_state.messages = []
                                                st.success("🤖 **Agentes LangGraph inicializados**")
                                            else:
                                                st.warning("⚠️ **Agentes no inicializados**")
                                                st.info("Los agentes requieren API key válida. Configúrala en Configuración.")
                                        else:
                                            st.error("❌ **Error en indexación**")
                                    except Exception as e:
                                        error_msg = str(e).lower()
                                        if 'quota' in error_msg or '429' in error_msg or 'rate limit' in error_msg:
                                            st.error("⚠️ **Límite de API alcanzado**")
                                            st.info("Has superado el límite diario de solicitudes. Las consultas estarán disponibles mañana.")
                                            st.session_state.daily_limit_reached = True
                                        else:
                                            st.error(f"❌ **Error:** {str(e)}")
                                            st.info("Verifica tu API key en la pestaña Configuración.")
                        else:
                            st.error("❌ **Error: Repositorio sin ID**")
                    else:
                        st.error("❌ **No se encontraron archivos válidos**")
            else:
                st.error("❌ **Directorio no válido**")


def show_chat_section() -> None:
    """Sección de chat con agentes LangGraph y manejo de límite de API."""
    st.title("💬 Chat con Agentes IA")
    
    if 'current_repo' not in st.session_state or not st.session_state.current_repo:
        st.warning("⚠️ **Primero carga un repositorio**")
        st.info("👉 Ve a la pestaña **Cargar Repositorio**")
        return
    
    if 'rag_service' not in st.session_state or not st.session_state.rag_service:
        st.warning("⚠️ **El repositorio no está indexado**")
        st.info("👉 Vuelve a cargar el repositorio para indexarlo")
        return
    
    if 'agent_workflow' not in st.session_state or not st.session_state.agent_workflow:
        st.warning("⚠️ **Los agentes no están disponibles**")
        st.info("Verifica que tu API key sea válida en la pestaña Configuración.")
        return
    
    # Verificar límite diario al inicio del chat
    quota_exceeded = _check_quota_and_display()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "agent_used" in message and message["role"] == "assistant":
                st.caption(f"🤖 Agente: {message['agent_used']}")
            if "sources" in message and message["sources"]:
                with st.expander("📚 Fuentes consultadas"):
                    for source in message["sources"]:
                        st.write(f"**📄 {source['file']}**")
                        
    
    if prompt := st.chat_input("💬 Pregunta sobre el código..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            # Verificar límite diario antes de procesar
            if quota_exceeded or st.session_state.get('daily_limit_reached', False):
                st.error("⚠️ **Límite diario de API alcanzado**")
                st.info("""
                Has superado el límite diario de solicitudes de Gemini API.
                
                **¿Qué puedes hacer?**
                - Esperar hasta mañana para que se reinicie el contador
                - Las consultas estarán disponibles nuevamente mañana
                
                **Mientras tanto:**
                - Puedes seguir navegando por repositorios ya indexados
                - Los análisis de código ya generados siguen disponibles
                """)
                response = "⚠️ **Límite diario de API alcanzado.** Las consultas estarán disponibles mañana. Por favor, intenta nuevamente más tarde."
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "agent_used": "Sistema",
                    "sources": []
                })
                return
            
            with st.spinner("🤖 Procesando con agentes..."):
                try:
                    result = st.session_state.agent_workflow.process(prompt)
                    
                    answer = result.get('answer', 'No se pudo generar respuesta')
                    sources = result.get('sources', [])
                    agent_used = result.get('agent', 'desconocido')
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📚 Fuentes consultadas"):
                            for source in sources[:3]:
                                st.write(f"**📄 {source['file']}**")
                    
                    st.caption(f"🤖 Procesado por: **{agent_used}**")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "agent_used": agent_used
                    })
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    # Detectar error de cuota (429)
                    if '429' in error_msg or 'quota' in error_msg or 'rate limit' in error_msg:
                        st.error("⚠️ **Límite de API alcanzado**")
                        st.info("""
                        Has superado el límite diario de solicitudes de Gemini API.
                        
                        **¿Qué puedes hacer?**
                        - Esperar hasta mañana para que se reinicie el contador (generalmente a las 00:00 PST)
                        - Las consultas estarán disponibles nuevamente mañana
                        
                        **Límites del free tier:**
                        - 20 solicitudes de texto por minuto
                        - 100 solicitudes de embeddings por minuto
                        - 1500 solicitudes por día (aproximadamente)
                        """)
                        st.session_state.daily_limit_reached = True
                        response = "⚠️ **Límite diario de API alcanzado.** Las consultas estarán disponibles mañana. Por favor, intenta nuevamente más tarde."
                    else:
                        st.error(f"❌ **Error:** {str(e)}")
                        response = f"Error: {str(e)}"
                    
                    st.markdown(response)
                    logger.error(f"Error en chat: {e}", exc_info=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": [],
                        "agent_used": "Error"
                    })


def show_analysis_section() -> None:
    """Sección de análisis con vista por lenguaje y soporte Django."""
    st.title("📊 Análisis del Repositorio")
    
    if 'current_repo' not in st.session_state or not st.session_state.current_repo:
        st.warning("⚠️ **Primero carga un repositorio**")
        st.info("👉 Ve a la pestaña **Cargar Repositorio** para comenzar")
        return
    
    repo = st.session_state.current_repo
    
    if isinstance(repo, dict):
        st.error("❌ **Error: Repositorio cargado incorrectamente**")
        st.info("Por favor, vuelve a cargar el repositorio desde la página de inicio o desde 'Cargar Repositorio'")
        return
    
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
    
    if hasattr(repo, 'metadata') and repo.metadata.get('framework') == 'django':
        st.success("🐍 **Proyecto Django detectado**")
        
        models_count = len([f for f in repo.files if 'models.py' in f.name])
        views_count = len([f for f in repo.files if 'views.py' in f.name])
        urls_count = len([f for f in repo.files if 'urls.py' in f.name])
        admin_count = len([f for f in repo.files if 'admin.py' in f.name])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📦 Modelos", models_count)
        with col2:
            st.metric("👁️ Vistas", views_count)
        with col3:
            st.metric("🔗 URLs", urls_count)
        with col4:
            st.metric("⚙️ Admin", admin_count)
    
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
            
            if file.functions:
                st.write("**🔧 Funciones:**")
                for func in file.functions[:10]:
                    if isinstance(func, dict):
                        func_name = func.get('name', 'unknown')
                        line_start = func.get('line_start', 0)
                        line_end = func.get('line_end', 0)
                    else:
                        func_name = func.name
                        line_start = func.line_start
                        line_end = func.line_end
                    st.write(f"- `{func_name}` ({line_start}-{line_end})")
            
            if file.classes:
                st.write("**📚 Clases:**")
                for cls in file.classes[:5]:
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
    """Muestra lista de repositorios con eliminación directa."""
    st.title("📚 Repositorios Analizados")
    
    if 'repo_service' not in st.session_state:
        st.session_state.repo_service = RepositoryService()
    
    repos = st.session_state.repo_service.list_repositories()
    
    if not repos:
        st.info("📭 **No hay repositorios analizados**")
        st.info("👉 Ve a la pestaña **Cargar Repositorio** para comenzar")
        return
    
    # Información sobre eliminación
    with st.expander("ℹ️ Información sobre eliminación", expanded=False):
        st.markdown("""
        **Al eliminar un repositorio se elimina:**
        - ✅ Metadatos de la base de datos
        - ✅ Índices de búsqueda (FAISS)
        - ✅ Fragmentos de código en caché
        - ✅ Copia del repositorio en `data/repositories/`
        
        **NO se elimina:**
        - ❌ El repositorio original en tu disco (si cargaste desde directorio local)
        
        **Para volver a cargar el repositorio:**
        - Solo necesitas volver a cargarlo desde la misma ubicación
        - Se regenerarán los índices automáticamente
        """)
    
    for repo in repos:
        with st.container():
            cols = st.columns([3, 1, 1, 1, 1])
            
            with cols[0]:
                st.write(f"**{repo['name']}**")
                path = Path(repo['path'])
                if path.exists():
                    if "data/repositories" in str(path):
                        st.caption(f"📁 Copia en caché")
                    else:
                        st.caption(f"📁 Original: {repo['path']}")
                else:
                    st.caption("⚠️ No encontrado en disco")
            
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
                if st.button("🗑️ Eliminar", key=f"delete_{repo['id']}", help="Eliminar repositorio del sistema"):
                    with st.spinner(f"Eliminando repositorio {repo['name']}..."):
                        success = st.session_state.repo_service.delete_repository(repo['id'])
                        if success:
                            st.success(f"✅ Repositorio '{repo['name']}' eliminado del sistema")
                            if 'current_repo' in st.session_state and st.session_state.current_repo:
                                if hasattr(st.session_state.current_repo, 'name') and \
                                   st.session_state.current_repo.name == repo['name']:
                                    st.session_state.current_repo = None
                                    st.session_state.repository_loaded = False
                                    st.session_state.rag_service = None
                                    st.session_state.agent_workflow = None
                                    st.session_state.messages = []
                            st.rerun()
                        else:
                            st.error(f"❌ Error al eliminar '{repo['name']}'")
                            st.info("Verifica que no haya archivos en uso o que tengas permisos suficientes.")
            
            st.divider()


def show_configuration_section() -> None:
    """Configuración avanzada."""
    st.title("⚙️ Configuración")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔑 API Key", "🤖 Modelos", "📁 Límites", "📊 Sistema"])
    
    with tab1:
        st.subheader("🔑 API Key de Gemini")
        
        current_key = os.getenv("GEMINI_API_KEY", "")
        is_configured = ServiceFactory.is_api_key_valid()
        
        if is_configured:
            masked = current_key[:10] + "..." + current_key[-5:] if current_key else "Configurada"
            st.success(f"✅ **API Key configurada:** `{masked}`")
        else:
            st.error("❌ **API Key no configurada**")
            st.info("""
            **Para obtener tu API key gratuita:**
            1. Ve a [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
            2. Inicia sesión con tu cuenta Google
            3. Crea una nueva API key
            4. Cópiala y pégala aquí
            """)
        
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
                if not new_key and not is_configured:
                    st.warning("Primero ingresa una API Key")
                else:
                    try:
                        test_key = new_key or current_key
                        if test_key:
                            import google.generativeai as genai
                            genai.configure(api_key=test_key)
                            models = genai.list_models()
                            st.success("✅ Conexión exitosa! API Key válida")
                        else:
                            st.error("No hay API Key para probar")
                    except Exception as e:
                        error_msg = str(e).lower()
                        if 'quota' in error_msg or '429' in error_msg:
                            st.error("⚠️ Límite de API alcanzado. Espera hasta mañana.")
                        elif 'api_key' in error_msg or 'invalid' in error_msg:
                            st.error("❌ API Key inválida. Verifica que la hayas copiado correctamente.")
                        else:
                            st.error(f"❌ Error: {e}")
    
    with tab2:
        st.subheader("🤖 Modelos")
        prefer_pro = st.radio(
            "Preferencia:",
            ["⚡ Flash (gratuito)", "⭐ Pro (más capaz)"],
            index=0 if not st.session_state.get('prefer_pro', False) else 1
        )
        st.session_state['prefer_pro'] = ("Pro" in prefer_pro)
        
        temperature = st.slider("🌡️ Temperatura:", 0.0, 1.0, st.session_state.get('temperature', 0.2), 0.1)
        st.session_state['temperature'] = temperature
        
        k_results = st.slider("📚 Fragmentos:", 1, 10, st.session_state.get('k_results', 5))
        st.session_state['k_results'] = k_results
    
    with tab3:
        st.subheader("📁 Límites")
        max_file_size = st.slider("📦 Tamaño máximo (MB):", 1, 10, st.session_state.get('max_file_size_mb', 1))
        st.session_state['max_file_size_mb'] = max_file_size
        
        include_docs = st.checkbox("📚 Incluir documentación", st.session_state.get('include_docs', False))
        st.session_state['include_docs'] = include_docs
        
        if st.button("💾 Guardar"):
            st.success("✅ Configuración guardada")
    
    with tab4:
        st.subheader("📊 Sistema")
        st.write(f"- Repositorios: `{Path('data/repositories').absolute()}`")
        st.write(f"- Vectores: `{Path('data/vectors').absolute()}`")
        st.write(f"- Caché: `{Path('data/cache').absolute()}`")
        
        if st.button("🧹 Limpiar vectores", use_container_width=True):
            import shutil
            vectors_dir = Path("data/vectors")
            if vectors_dir.exists():
                shutil.rmtree(vectors_dir)
                vectors_dir.mkdir()
                st.success("✅ Vectores limpiados")
                st.rerun()