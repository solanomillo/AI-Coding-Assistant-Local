"""
Módulo de interfaz de usuario para AI Coding Assistant Local.
ARQUITECTURA CORREGIDA - FASE 2/3:
- Upload con persistencia permanente
- Integración con repo_service y rag_service
- Manejo robusto de errores
"""

import streamlit as st
import logging
from pathlib import Path
import tempfile
import os
from dotenv import load_dotenv, set_key
from infrastructure.llm_clients.gemini_llm import GeminiLLM
from application.services.repo_service import RepositoryService
from application.services.rag_gemini_service import RAGGeminiService

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


def show_upload_section() -> None:
    """Sección de carga con debug visible."""
    st.title("📤 Cargar Repositorio")
    
    if 'repo_service' not in st.session_state:
        st.session_state.repo_service = RepositoryService()
    
    # Área de debug
    debug_expander = st.expander("🔍 Ver debug", expanded=True)
    
    upload_method = st.radio(
        "Método de carga:",
        ["Archivo ZIP", "Directorio local"],
        horizontal=True
    )
    
    if upload_method == "Archivo ZIP":
        uploaded_file = st.file_uploader(
            "Selecciona un archivo ZIP",
            type=['zip']
        )
        
        if uploaded_file:
            with debug_expander:
                st.write("📦 **Información del ZIP:**")
                st.write(f"- Nombre: {uploaded_file.name}")
                st.write(f"- Tamaño: {len(uploaded_file.getvalue()) / 1024:.2f} KB")
            
            # Guardar ZIP temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            with st.spinner("📁 Procesando repositorio..."):
                try:
                    # Cargar desde ZIP
                    repo = st.session_state.repo_service.load_from_zip(tmp_path)
                    
                    with debug_expander:
                        st.write("📁 **Resultado del análisis:**")
                        if repo:
                            st.write(f"✅ Repositorio: {repo.name}")
                            st.write(f"📁 Ruta física: {repo.path}")
                            st.write(f"📄 Archivos Python: {len(repo.files)}")
                            
                            # Mostrar primeros archivos con sus rutas relativas
                            st.write("**Primeros archivos:**")
                            for f in repo.files[:5]:
                                st.write(f"  - {f.relative_path}")
                        else:
                            st.error("❌ No se pudo analizar el repositorio")
                    
                    if repo and repo.files:
                        st.success(f"✅ Repositorio '{repo.name}' procesado")
                        
                        # Indexar
                        with st.spinner("🧠 Indexando con Gemini..."):
                            try:
                                rag_service = RAGGeminiService(
                                    repo_name=repo.name,
                                    repo_path=repo.path,
                                    prefer_pro=st.session_state.get('prefer_pro', False)
                                )
                                
                                if rag_service.index_repository(repo):
                                    st.success("✅ Indexación completada")
                                    st.session_state.rag_service = rag_service
                                    st.session_state.repository_loaded = True
                                    st.session_state.current_repo = repo
                                    
                                    stats = rag_service.get_stats()
                                    st.metric("Fragmentos indexados", stats['vector_store']['total_vectors'])
                                else:
                                    st.error("❌ Error en indexación")
                            except Exception as e:
                                st.error(f"Error en Gemini: {str(e)}")
                                with debug_expander:
                                    st.exception(e)
                    else:
                        st.error("❌ No se encontraron archivos Python")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    with debug_expander:
                        st.exception(e)
                finally:
                    os.unlink(tmp_path)


def show_chat_section() -> None:
    """Sección de chat con el repositorio."""
    st.title("💬 Chat con el Repositorio")
    
    if 'current_repo' not in st.session_state or not st.session_state.current_repo:
        st.warning("Primero carga un repositorio")
        return
    
    if 'rag_service' not in st.session_state:
        st.warning("El repositorio no está indexado. Vuelve a cargarlo.")
        return
    
    # Sidebar con información
    with st.sidebar:
        st.info(f"📚 Repositorio: **{st.session_state.current_repo.name}**")
        
        stats = st.session_state.rag_service.get_stats()
        st.metric("Fragmentos indexados", stats['vector_store']['total_vectors'])
        
        # Información del modelo
        model_info = stats['llm']
        model_emoji = "⭐ PRO" if model_info['model_type'] == 'pro' else "⚡ FLASH"
        st.success(f"**Modelo activo:** {model_emoji}")
        st.caption(f"`{model_info['current_model']}`")
        
        # Opción para cambiar modelo
        current_pref = st.session_state.get('prefer_pro', False)
        new_pref = st.checkbox(
            "✨ Preferir modelo Pro", 
            value=current_pref,
            help="Modelos Pro más capaces pero con rate limits"
        )
        
        if new_pref != current_pref:
            st.session_state['prefer_pro'] = new_pref
            st.rerun()
    
    # Historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "model_used" in message and message["role"] == "assistant":
                st.caption(f"🤖 {message['model_used']}")
            if "sources" in message and message["sources"]:
                with st.expander("📚 Fuentes"):
                    for source in message["sources"]:
                        st.write(f"**{source['file']}** (score: {source.get('score', 'N/A')})")
                        st.code(source['preview'], language='python')
    
    # Input
    if prompt := st.chat_input("Pregunta sobre el código..."):
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
                        k=5,
                        include_sources=True,
                        temperature=st.session_state.get('temperature', 0.2)
                    )
                    
                    answer = result['answer']
                    sources = result['sources']
                    model_used = result.get('model_used', 'desconocido')
                    
                    st.markdown(answer)
                    st.caption(f"🤖 Usando: `{model_used}`")
                    
                    if sources:
                        with st.expander("📚 Fuentes consultadas"):
                            for source in sources:
                                st.write(f"**{source['file']}**")
                                st.code(source['preview'], language='python')
                    
                    # Guardar en historial
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "model_used": model_used
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Error en chat: {e}", exc_info=True)


def show_repositories_list() -> None:
    """Muestra lista de repositorios con estado de archivos."""
    st.title("📚 Repositorios Analizados")
    
    if 'repo_service' not in st.session_state:
        st.session_state.repo_service = RepositoryService()
    
    repos = st.session_state.repo_service.list_repositories()
    
    if not repos:
        st.info("No hay repositorios analizados. Ve a 'Cargar Repositorio' para comenzar.")
        return
    
    for repo in repos:
        with st.container():
            cols = st.columns([3, 1, 1, 1, 1])
            
            with cols[0]:
                st.write(f"**{repo['name']}**")
                # Mostrar si los archivos existen
                if repo.get('files_exist', False):
                    st.caption(f"✅ Archivos en disco ({repo.get('size_mb', 0)} MB)")
                else:
                    st.caption("⚠️ Archivos no encontrados en disco")
            
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
                if st.button("Cargar", key=f"load_{repo['id']}"):
                    st.info("Funcionalidad de carga desde BD en desarrollo")
            
            st.divider()


def show_configuration_section() -> None:
    """Configuración de Gemini."""
    st.title("⚙️ Configuración")
    
    tab1, tab2, tab3 = st.tabs(["🔑 API Key", "🤖 Modelos", "📁 Sistema"])
    
    with tab1:
        st.subheader("API Key de Gemini")
        
        current_key = os.getenv("GEMINI_API_KEY", "")
        masked = current_key[:10] + "..." + current_key[-5:] if current_key else "No configurada"
        
        st.info(f"API Key actual: `{masked}`")
        
        new_key = st.text_input(
            "Nueva API Key:",
            type="password",
            placeholder="AIzaSy..."
        )
        
        if st.button("Guardar API Key"):
            if new_key:
                if save_env_var("GEMINI_API_KEY", new_key):
                    st.success("✅ API Key guardada")
                    st.rerun()
                else:
                    st.error("Error guardando API Key")
        
        if st.button("Probar conexión"):
            try:                
                test = GeminiLLM()
                models = test.list_available_models()
                st.success(f"✅ Conexión exitosa! {len(models)} modelos disponibles")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    with tab2:
        st.subheader("Configuración de Modelos")
        
        prefer_pro = st.radio(
            "Preferencia de modelo:",
            ["⚡ Flash (gratuito, rápido)", "⭐ Pro (más capaz)"],
            index=0 if not st.session_state.get('prefer_pro', False) else 1
        )
        
        st.session_state['prefer_pro'] = ("Pro" in prefer_pro)
        
        temperature = st.slider(
            "Temperatura:",
            0.0, 1.0, st.session_state.get('temperature', 0.2), 0.1
        )
        st.session_state['temperature'] = temperature
        
        k_results = st.slider(
            "Fragmentos a recuperar:",
            1, 20, st.session_state.get('k_results', 5)
        )
        st.session_state['k_results'] = k_results
    
    with tab3:
        st.subheader("Información del Sistema")
        
        # Mostrar directorios
        st.write("**Directorios de datos:**")
        st.write(f"- 📁 Repositorios: `{Path('data/repositories').absolute()}`")
        st.write(f"- 🔍 Vector store: `{Path('data/vector_store').absolute()}`")
        
        # Estadísticas
        if 'rag_service' in st.session_state:
            stats = st.session_state.rag_service.get_stats()
            st.json({
                'repo_actual': stats['repo_name'],
                'fragmentos': stats['vector_store']['total_vectors'],
                'modelo': stats['llm']['current_model']
            })
        
        # Botón de limpieza
        if st.button("🧹 Limpiar caché de vectores"):
            import shutil
            cache_dir = Path("data/vector_store")
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                cache_dir.mkdir()
                st.success("✅ Caché limpiado")