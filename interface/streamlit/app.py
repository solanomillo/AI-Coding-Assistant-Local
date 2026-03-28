"""
Módulo de interfaz de usuario para AI Coding Assistant Local.
"""

import streamlit as st
import logging
from pathlib import Path
import tempfile
import os
from dotenv import load_dotenv, set_key
from typing import Union, Dict, Any, List, Tuple

from application.services.repo_service import RepositoryService
from application.services.service_factory import ServiceFactory
from application.graph.workflow import AgentWorkflow

logger = logging.getLogger(__name__)

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


def get_repo_name(repo: Union[Dict[str, Any], Any]) -> str:
    """Obtiene el nombre del repositorio."""
    if isinstance(repo, dict):
        return repo.get('name', 'desconocido')
    return getattr(repo, 'name', 'desconocido')


def get_repo_file_count(repo: Union[Dict[str, Any], Any]) -> int:
    """Obtiene el numero de archivos."""
    if isinstance(repo, dict):
        return repo.get('file_count', 0)
    if hasattr(repo, 'files'):
        return len(repo.files)
    return getattr(repo, 'file_count', 0)


def _check_quota(model_name: str = None, force_check: bool = False) -> Tuple[bool, str, str]:
    """
    Verifica si la cuota esta disponible.
    Retorna (available, status, message)
    """
    if model_name is None:
        model_name = st.session_state.get('selected_model', 'gemini-2.5-flash')
    
    available, status = ServiceFactory.check_quota_available(model_name, force_check=force_check)
    
    if available:
        return True, "OK", ""
    
    if status == "API_KEY_NOT_CONFIGURED":
        return False, "API_KEY_NOT_CONFIGURED", "🔑 API key no configurada"
    
    if status == "QUOTA_EXCEEDED":
        return False, "QUOTA_EXCEEDED", "⚠️ Límite diario de API alcanzado. Las consultas estarán disponibles mañana."
    
    if "MODELO_NO_DISPONIBLE" in status:
        return False, "MODELO_NO_DISPONIBLE", f"❌ El modelo '{model_name}' no está disponible con tu API key"
    
    if status == "INVALID_API_KEY":
        return False, "INVALID_API_KEY", "❌ API key inválida. Verifica que la hayas copiado correctamente."
    
    return False, "ERROR", f"❌ Error de API: {status}"


def _show_quota_warning() -> None:
    """Muestra advertencia de cuota agotada en la interfaz."""
    st.error("⚠️ **Límite diario de API alcanzado**")
    st.info("""
    Has superado el límite diario de solicitudes de Gemini API.
    
    **¿Qué puedes hacer?**
    - Esperar hasta mañana para que se reinicie el contador
    - Las consultas estarán disponibles nuevamente mañana
    - Puedes seguir usando repositorios ya indexados para consultas
    - No se pueden indexar nuevos repositorios hasta mañana
    
    **Límites del free tier:**
    - 20 solicitudes de texto por minuto
    - 100 solicitudes de embeddings por minuto
    - 1500 solicitudes por día (aproximadamente)
    """)
    st.session_state.daily_limit_reached = True


def _show_api_key_warning() -> None:
    """Muestra advertencia de API key no configurada."""
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


def _setup_services(repo) -> Tuple[bool, Any, Any, str]:
    """
    Configura los servicios para un repositorio.
    Retorna (success, rag_service, agent_workflow, message)
    """
    # Verificar cuota primero
    available, status, message = _check_quota()
    
    if not available:
        if status == "QUOTA_EXCEEDED":
            return False, None, None, "⚠️ Límite diario de API alcanzado. No se puede crear el servicio RAG hasta mañana."
        elif status == "API_KEY_NOT_CONFIGURED":
            return False, None, None, "🔑 API Key no configurada. Ve a Configuración para agregarla."
        else:
            return False, None, None, f"❌ Error: {message}"
    
    rag_service, agent_workflow = ServiceFactory.setup_repository_services(
        repo=repo,
        model_name=st.session_state.get('selected_model', 'gemini-2.5-flash'),
        max_file_size_mb=st.session_state.get('max_file_size_mb', 1),
        include_docs=st.session_state.get('include_docs', False)
    )
    
    if rag_service is None:
        return False, None, None, "❌ No se pudo crear el servicio RAG. Verifica tu API key y conexión."
    
    if agent_workflow is None:
        return False, None, None, "❌ No se pudo crear el flujo de agentes."
    
    return True, rag_service, agent_workflow, ""


def _show_repository_stats(repo) -> None:
    """Muestra estadisticas del repositorio."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Archivos", len(repo.files))
    with col2:
        total_funcs = sum(len(f.functions) for f in repo.files)
        st.metric("🔧 Funciones", total_funcs)
    with col3:
        total_classes = sum(len(f.classes) for f in repo.files)
        st.metric("📚 Clases", total_classes)


def show_upload_section() -> None:
    """Seccion de carga de repositorios."""
    st.title("📤 Cargar Repositorio")
    
    if 'repo_service' not in st.session_state:
        st.session_state.repo_service = RepositoryService()
    
    available, status, message = _check_quota()
    
    if not available:
        if status == "API_KEY_NOT_CONFIGURED":
            _show_api_key_warning()
        elif status == "QUOTA_EXCEEDED":
            _show_quota_warning()
        else:
            st.error(message)
        return
    
    with st.expander("⚡ Optimizaciones activas", expanded=False):
        st.markdown("""
        - 📄 Máximo **10 fragmentos por archivo**
        - 📏 Fragmentos de **500 caracteres**
        - 💾 Archivos mayores a **1 MB** son ignorados
        - 📊 Archivos con más de **2000 líneas** son ignorados
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
            original_filename = uploaded_file.name
            st.info(f"📄 **Archivo subido:** {original_filename}")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            with st.spinner("📁 Procesando repositorio..."):
                try:
                    repo = st.session_state.repo_service.load_from_zip_with_name(tmp_path, original_filename)
                    
                    if repo:
                        if hasattr(repo, 'db_id') and repo.db_id:
                            safe_name = repo.name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                            index_path = Path(f"data/vectors/{safe_name}.index")
                            
                            if index_path.exists():
                                st.warning("⚠️ **Repositorio ya existente**")
                                st.info("📦 Cargado desde caché.")
                                
                                success, rag, agent, error_msg = _setup_services(repo)
                                if not success:
                                    st.error(error_msg)
                                    return
                                
                                if success:
                                    st.session_state.rag_service = rag
                                    st.session_state.agent_workflow = agent
                                    st.session_state.current_repo = repo
                                    st.session_state.repository_loaded = True
                                    st.session_state.messages = []
                                    st.success("✅ Repositorio cargado correctamente")
                                else:
                                    st.error("❌ Error al configurar servicios")
                            else:
                                st.success(f"✅ **Repositorio '{repo.name}' procesado correctamente**")
                                _show_repository_stats(repo)
                                
                                avail, stat, msg = _check_quota()
                                if not avail:
                                    if stat == "QUOTA_EXCEEDED":
                                        _show_quota_warning()
                                    else:
                                        st.error(msg)
                                    return
                                
                                with st.spinner("🧠 Indexando..."):
                                    rag = ServiceFactory.create_rag_service(
                                        repo_name=repo.name,
                                        repo_path=repo.path,
                                        repo_id=repo.db_id,
                                        model_name=st.session_state.get('selected_model', 'gemini-2.5-flash'),
                                        max_file_size_mb=st.session_state.get('max_file_size_mb', 1),
                                        include_docs=st.session_state.get('include_docs', False)
                                    )
                                    
                                    if rag is None:
                                        st.error("❌ No se pudo crear el servicio RAG. Verifica tu API key y cuota.")
                                        return
                                    
                                    if rag.index_repository(repo):
                                        st.success("✅ **Indexación completada**")
                                        success, rag, agent, error_msg = _setup_services(repo)
                                        if not success:
                                            st.error(error_msg)
                                            return
                                        
                                        if success:
                                            st.session_state.rag_service = rag
                                            st.session_state.agent_workflow = agent
                                            st.session_state.current_repo = repo
                                            st.session_state.repository_loaded = True
                                            st.session_state.messages = []
                                            st.success("🤖 **Agentes inicializados**")
                                    else:
                                        st.error("❌ Error en indexación")
                    else:
                        st.error("❌ No se encontraron archivos válidos")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                finally:
                    os.unlink(tmp_path)
    
    else:
        repo_path = st.text_input(
            "📁 Ruta del directorio:",
            placeholder="C:/Users/usuario/mi-repositorio"
        )
        
        if repo_path:
            path = Path(repo_path)
            if path.exists() and path.is_dir():
                with st.spinner("📁 Analizando repositorio..."):
                    repo = st.session_state.repo_service.load_from_directory(path)
                    
                    if repo:
                        if hasattr(repo, 'db_id') and repo.db_id:
                            safe_name = repo.name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                            index_path = Path(f"data/vectors/{safe_name}.index")
                            
                            if index_path.exists():
                                st.warning("⚠️ **Repositorio ya existente**")
                                
                                success, rag, agent, error_msg = _setup_services(repo)
                                if not success:
                                    st.error(error_msg)
                                    return
                                
                                if success:
                                    st.session_state.rag_service = rag
                                    st.session_state.agent_workflow = agent
                                    st.session_state.current_repo = repo
                                    st.session_state.repository_loaded = True
                                    st.session_state.messages = []
                                    st.success("✅ Repositorio cargado correctamente")
                                else:
                                    st.error("❌ Error al configurar servicios")
                            else:
                                st.success(f"✅ **Repositorio '{repo.name}' procesado correctamente**")
                                _show_repository_stats(repo)
                                
                                avail, stat, msg = _check_quota()
                                if not avail:
                                    if stat == "QUOTA_EXCEEDED":
                                        _show_quota_warning()
                                    else:
                                        st.error(msg)
                                    return
                                
                                with st.spinner("🧠 Indexando..."):
                                    rag = ServiceFactory.create_rag_service(
                                        repo_name=repo.name,
                                        repo_path=path,
                                        repo_id=repo.db_id,
                                        model_name=st.session_state.get('selected_model', 'gemini-2.5-flash'),
                                        max_file_size_mb=st.session_state.get('max_file_size_mb', 1),
                                        include_docs=st.session_state.get('include_docs', False)
                                    )
                                    
                                    if rag is None:
                                        st.error("❌ No se pudo crear el servicio RAG. Verifica tu API key y cuota.")
                                        return
                                    
                                    if rag.index_repository(repo):
                                        st.success("✅ **Indexación completada**")
                                        success, rag, agent, error_msg = _setup_services(repo)
                                        if not success:
                                            st.error(error_msg)
                                            return
                                        
                                        if success:
                                            st.session_state.rag_service = rag
                                            st.session_state.agent_workflow = agent
                                            st.session_state.current_repo = repo
                                            st.session_state.repository_loaded = True
                                            st.session_state.messages = []
                                            st.success("🤖 **Agentes inicializados**")
                                    else:
                                        st.error("❌ Error en indexación")
                    else:
                        st.error("❌ No se encontraron archivos válidos")
            else:
                st.error("❌ Directorio no válido")


def show_chat_section() -> None:
    """Seccion de chat con agentes."""
    st.title("💬 Chat con Agentes IA")
    
    if 'current_repo' not in st.session_state or not st.session_state.current_repo:
        st.warning("⚠️ **Primero carga un repositorio**")
        return
    
    if 'rag_service' not in st.session_state or not st.session_state.rag_service:
        st.warning("⚠️ **El repositorio no está indexado**")
        return
    
    if 'agent_workflow' not in st.session_state or not st.session_state.agent_workflow:
        st.warning("⚠️ **Los agentes no están disponibles**")
        return
    
    available, status, message = _check_quota()
    
    if not available:
        if status == "QUOTA_EXCEEDED":
            _show_quota_warning()
            st.info("💡 Puedes seguir usando repositorios ya indexados para consultas, pero no se generarán nuevas respuestas hasta que se restablezca la cuota.")
        elif status == "API_KEY_NOT_CONFIGURED":
            _show_api_key_warning()
        else:
            st.error(message)
        return
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "agent_used" in message:
                st.caption(f"🤖 Agente: {message['agent_used']}")
            if "sources" in message and message["sources"]:
                with st.expander("📚 Fuentes consultadas"):
                    for source in message["sources"]:
                        st.write(f"📄 **{source['file']}**")
    
    if prompt := st.chat_input("💬 Pregunta sobre el código..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤖 Procesando..."):
                try:
                    result = st.session_state.agent_workflow.process(prompt)
                    
                    answer = result.get('answer', 'No se pudo generar respuesta')
                    sources = result.get('sources', [])
                    agent_used = result.get('agent', 'desconocido')
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📚 Fuentes consultadas"):
                            for source in sources[:5]:
                                st.write(f"📄 **{source['file']}**")
                    
                    st.caption(f"🤖 Procesado por: **{agent_used}**")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "agent_used": agent_used
                    })
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if 'quota' in error_msg or '429' in error_msg:
                        st.error("⚠️ **Límite de API alcanzado**")
                        st.info("Las consultas estarán disponibles mañana.")
                        st.session_state.daily_limit_reached = True
                    else:
                        st.error(f"❌ Error: {str(e)}")


def show_analysis_section() -> None:
    """Seccion de analisis del repositorio."""
    st.title("📊 Análisis del Repositorio")
    
    if 'current_repo' not in st.session_state or not st.session_state.current_repo:
        st.warning("⚠️ **Primero carga un repositorio**")
        return
    
    repo = st.session_state.current_repo
    
    if isinstance(repo, dict):
        st.error("❌ Error: Repositorio cargado incorrectamente")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Archivos", len(repo.files))
    with col2:
        st.metric("🔧 Funciones", sum(len(f.functions) for f in repo.files))
    with col3:
        st.metric("📚 Clases", sum(len(f.classes) for f in repo.files))
    with col4:
        st.metric("📊 Líneas", repo.total_lines)
    
    if hasattr(repo, 'metadata') and repo.metadata.get('framework') == 'django':
        st.success("🐍 **Proyecto Django detectado**")
        models = len([f for f in repo.files if 'models.py' in f.name])
        views = len([f for f in repo.files if 'views.py' in f.name])
        urls = len([f for f in repo.files if 'urls.py' in f.name])
        col1, col2, col3 = st.columns(3)
        col1.metric("📦 Modelos", models)
        col2.metric("👁️ Vistas", views)
        col3.metric("🔗 URLs", urls)
    
    extension_counts = {}
    for file in repo.files:
        ext = file.extension
        extension_counts[ext] = extension_counts.get(ext, 0) + 1
    
    if extension_counts:
        st.subheader("📊 Distribución por extensión")
        cols = st.columns(min(len(extension_counts), 5))
        for idx, (ext, count) in enumerate(sorted(extension_counts.items(), key=lambda x: x[1], reverse=True)):
            cols[idx % 5].metric(ext, count)
    
    st.subheader("📁 Archivos del repositorio")
    
    extensions = sorted(set(f.extension for f in repo.files))
    selected_ext = st.selectbox("Filtrar por extensión:", ["Todos"] + extensions)
    
    files_to_show = repo.files
    if selected_ext != "Todos":
        files_to_show = [f for f in repo.files if f.extension == selected_ext]
    
    for file in files_to_show[:50]:
        with st.expander(f"📄 {file.name}"):
            col1, col2, col3 = st.columns(3)
            col1.write(f"**Líneas:** {file.line_count}")
            col2.write(f"**Funciones:** {len(file.functions)}")
            col3.write(f"**Clases:** {len(file.classes)}")
            
            if file.functions:
                st.write("**🔧 Funciones:**")
                for func in file.functions[:10]:
                    name = func.name if hasattr(func, 'name') else func.get('name', 'unknown')
                    start = func.line_start if hasattr(func, 'line_start') else func.get('line_start', 0)
                    end = func.line_end if hasattr(func, 'line_end') else func.get('line_end', 0)
                    st.write(f"- `{name}` ({start}-{end})")
            
            if file.classes:
                st.write("**📚 Clases:**")
                for cls in file.classes[:5]:
                    name = cls.name if hasattr(cls, 'name') else cls.get('name', 'unknown')
                    methods = cls.methods if hasattr(cls, 'methods') else cls.get('methods', [])
                    st.write(f"- `{name}`")
                    for method in methods[:3]:
                        m_name = method.name if hasattr(method, 'name') else method.get('name', 'unknown')
                        st.write(f"  - método: `{m_name}()`")


def show_repositories_list() -> None:
    """Muestra lista de repositorios."""
    st.title("📚 Repositorios Analizados")
    
    if 'repo_service' not in st.session_state:
        st.session_state.repo_service = RepositoryService()
    
    repos = st.session_state.repo_service.list_repositories()
    
    if not repos:
        st.info("📭 No hay repositorios analizados")
        return
    
    for repo in repos:
        with st.container():
            cols = st.columns([3, 1, 1, 1, 1])
            cols[0].write(f"**{repo['name']}**")
            path = Path(repo['path'])
            if path.exists():
                if "data/repositories" in str(path):
                    cols[0].caption("📁 Copia en caché")
                else:
                    cols[0].caption(f"📁 Original: {repo['path']}")
            else:
                cols[0].caption("⚠️ No encontrado en disco")
            cols[1].write(f"📁 {repo['file_count']}")
            cols[2].write(f"📊 {repo['total_lines']}")
            created = repo['created_at']
            fecha = created.strftime("%Y-%m-%d") if hasattr(created, 'strftime') else str(created)[:10]
            cols[3].write(f"🕐 {fecha}")
            if cols[4].button("🗑️ Eliminar", key=f"delete_{repo['id']}"):
                if st.session_state.repo_service.delete_repository(repo['id']):
                    st.success(f"✅ Repositorio {repo['name']} eliminado")
                    if 'current_repo' in st.session_state and st.session_state.current_repo:
                        if get_repo_name(st.session_state.current_repo) == repo['name']:
                            st.session_state.current_repo = None
                            st.session_state.repository_loaded = False
                            st.session_state.rag_service = None
                            st.session_state.agent_workflow = None
                            st.session_state.messages = []
                    st.rerun()
                else:
                    st.error("❌ Error al eliminar")
            st.divider()


def show_configuration_section() -> None:
    """Configuracion con selector de modelos."""
    st.title("⚙️ Configuración")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔑 API Key", "🤖 Modelos", "📁 Límites", "📊 Sistema"])
    
    with tab1:
        st.subheader("🔑 API Key de Gemini")
        current_key = os.getenv("GEMINI_API_KEY", "")
        if current_key and current_key != "tu_api_key_aqui":
            st.success(f"✅ API Key configurada: `{current_key[:10]}...{current_key[-5:]}`")
        else:
            st.error("❌ API Key no configurada")
        
        new_key = st.text_input("Nueva API Key:", type="password", placeholder="AIzaSy...")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Guardar", use_container_width=True):
                if new_key:
                    if save_env_var("GEMINI_API_KEY", new_key):
                        st.success("✅ API Key guardada")
                        ServiceFactory.clear_cache()
                        st.rerun()
        with col2:
            if st.button("🔄 Probar conexión", use_container_width=True):
                selected_model = st.session_state.get('selected_model', 'gemini-2.5-flash')
                with st.spinner(f"Probando conexión con modelo {selected_model}..."):
                    available, status = ServiceFactory.check_quota_available(selected_model, force_check=True)
                    if available:
                        st.success(f"✅ Conexión exitosa con modelo: {selected_model}")
                    elif status == "QUOTA_EXCEEDED":
                        st.warning("⚠️ Cuota agotada. Espera hasta mañana.")
                    elif "MODELO_NO_DISPONIBLE" in status:
                        st.error(f"❌ El modelo '{selected_model}' no está disponible con tu API key")
                        st.info("Prueba seleccionando otro modelo en la pestaña 'Modelos'")
                    elif status == "INVALID_API_KEY":
                        st.error("❌ API Key inválida. Verifica que la hayas copiado correctamente.")
                    else:
                        st.error(f"❌ Error: {status}")
    
    with tab2:
        st.subheader("🤖 Selección de Modelo")
        
        if 'available_models' not in st.session_state:
            with st.spinner("Cargando modelos disponibles..."):
                st.session_state.available_models = ServiceFactory.get_available_models()
        
        models = st.session_state.available_models
        
        if models:
            model_options = {}
            for m in models:
                icon = "⚡" if m['type'] == 'flash' else "⭐"
                model_options[m['name']] = f"{icon} {m['display_name']}"
            
            current_model = st.session_state.get('selected_model', 'gemini-2.5-flash')
            if current_model not in model_options:
                current_model = 'gemini-2.5-flash'
            
            selected = st.selectbox(
                "Modelo a utilizar:",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
                index=list(model_options.keys()).index(current_model) if current_model in model_options else 0,
                help="Selecciona el modelo para generación de texto. Los modelos Flash son gratuitos, los Pro pueden tener costo."
            )
            
            st.session_state['selected_model'] = selected
            
            selected_info = next((m for m in models if m['name'] == selected), None)
            if selected_info:
                if selected_info['type'] == 'flash':
                    st.info("⚡ **Modelo Flash** - Gratuito, rápido y eficiente. Ideal para la mayoría de tareas.")
                else:
                    st.warning("⭐ **Modelo Pro** - Mayor capacidad pero puede consumir cuota más rápido. Verifica tus límites.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Probar este modelo", use_container_width=True):
                    with st.spinner(f"Probando modelo {selected}..."):
                        available, status = ServiceFactory.check_quota_available(selected, force_check=True)
                        if available:
                            st.success(f"✅ Modelo {selected} funciona correctamente")
                        elif "MODELO_NO_DISPONIBLE" in status:
                            st.error(f"❌ El modelo {selected} no está disponible con tu API key")
                            st.info("Prueba con otro modelo o verifica tu API key")
                        elif status == "QUOTA_EXCEEDED":
                            st.warning("⚠️ Cuota agotada. Espera hasta mañana.")
                        else:
                            st.error(f"❌ Error: {status}")
            
            with col2:
                if st.button("🔄 Recargar modelos", use_container_width=True):
                    with st.spinner("Consultando API..."):
                        st.session_state.available_models = ServiceFactory.get_available_models(force_refresh=True)
                        st.rerun()
        else:
            st.error("❌ No se pudieron cargar los modelos. Verifica tu conexión a internet.")
    
    with tab3:
        st.subheader("📁 Límites de Procesamiento")
        max_file_size = st.slider("📦 Tamaño máximo (MB):", 1, 10, st.session_state.get('max_file_size_mb', 1))
        st.session_state['max_file_size_mb'] = max_file_size
        
        include_docs = st.checkbox("📚 Incluir documentación", st.session_state.get('include_docs', False))
        st.session_state['include_docs'] = include_docs
        
        if st.button("💾 Guardar"):
            st.success("✅ Configuración guardada")
    
    with tab4:
        st.subheader("📊 Sistema")
        st.write(f"- 📁 Repositorios: `{Path('data/repositories').absolute()}`")
        st.write(f"- 🔍 Vectores: `{Path('data/vectors').absolute()}`")
        st.write(f"- 🗄️ Caché: `{Path('data/cache').absolute()}`")
        
        if st.button("🧹 Limpiar vectores", use_container_width=True):
            import shutil
            vectors_dir = Path("data/vectors")
            if vectors_dir.exists():
                shutil.rmtree(vectors_dir)
                vectors_dir.mkdir()
                st.success("✅ Vectores limpiados")
                st.rerun()