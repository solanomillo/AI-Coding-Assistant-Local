"""
Módulo principal del proyecto AI Coding Assistant Local.

Este archivo sirve como punto de entrada para la aplicación,
configurando el logging y la ejecución inicial del sistema.
"""

import logging
import sys
from pathlib import Path
from typing import NoReturn, Union, Dict, Any
import streamlit as st
from datetime import datetime
import os
from dotenv import load_dotenv
from application.services.rag_gemini_service import RAGService
from application.graph.workflow import AgentWorkflow

# Cargar variables de entorno
load_dotenv()

# Configurar logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ai_coding_assistant.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


def setup_project_structure() -> None:
    """
    Verifica y crea la estructura de directorios necesaria para el proyecto.
    """
    logger.info("Verificando estructura de directorios...")
    
    required_dirs = [
        "data/repositories",
        "data/vectors",
        "data/cache",
        "data/cache/files",
        "data/cache/metadata",
        "data/temp",
        "interface/streamlit",
        "application/agents",
        "application/services",
        "application/graph",
        "domain/models",
        "infrastructure/embeddings",
        "infrastructure/vector_db",
        "infrastructure/llm",
        "infrastructure/database",
        "infrastructure/parsers",
        "scripts",
        "logs"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directorio creado: {dir_path}")
        else:
            logger.debug(f"Directorio existente: {dir_path}")
    
    logger.info("Estructura de directorios verificada")


def get_repo_name(repo: Union[Dict[str, Any], Any]) -> str:
    """Obtiene el nombre del repositorio de manera segura."""
    if isinstance(repo, dict):
        return repo.get('name', 'desconocido')
    else:
        return getattr(repo, 'name', 'desconocido')


def get_repo_file_count(repo: Union[Dict[str, Any], Any]) -> int:
    """Obtiene el número de archivos del repositorio de manera segura."""
    if isinstance(repo, dict):
        return repo.get('file_count', 0)
    else:
        if hasattr(repo, 'files'):
            return len(repo.files)
        return getattr(repo, 'file_count', 0)


def get_repo_total_lines(repo: Union[Dict[str, Any], Any]) -> int:
    """Obtiene el total de líneas del repositorio de manera segura."""
    if isinstance(repo, dict):
        return repo.get('total_lines', 0)
    else:
        return getattr(repo, 'total_lines', 0)


def initialize_session_state() -> None:
    """
    Inicializa las variables de estado de la sesión de Streamlit.
    """
    if 'repository_loaded' not in st.session_state:
        st.session_state.repository_loaded = False
    
    if 'current_repo' not in st.session_state:
        st.session_state.current_repo = None
    
    if 'repo_service' not in st.session_state:
        from application.services.repo_service import RepositoryService
        st.session_state.repo_service = RepositoryService()
        logger.info("Servicio de repositorios inicializado")
    
    if 'rag_service' not in st.session_state:
        st.session_state.rag_service = None
    
    if 'agent_workflow' not in st.session_state:
        st.session_state.agent_workflow = None
    
    if 'prefer_pro' not in st.session_state:
        prefer_pro = os.getenv("GEMINI_PREFER_PRO", "false").lower() == "true"
        st.session_state.prefer_pro = prefer_pro
    
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
    
    if 'daily_limit_reached' not in st.session_state:
        st.session_state.daily_limit_reached = False


def show_welcome_page() -> None:
    """
    Muestra la página de bienvenida con información del sistema.
    """
    st.title("🤖 AI Coding Assistant Local")
    st.markdown("---")
    
    if st.session_state.get('daily_limit_reached', False):
        st.error("⚠️ **Límite diario de API alcanzado**")
        st.info("Las consultas estarán disponibles mañana. Puedes seguir usando repositorios ya indexados.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("📁 **Analiza repositorios**")
        st.markdown("""
        - Carga repositorios vía ZIP o directorio local
        - Analiza estructura de código automáticamente
        - Soporta **Python, JavaScript, TypeScript, HTML, CSS**
        - Extrae funciones, clases y metadatos
        """)
    
    with col2:
        st.info("🔍 **RAG System (Gemini + FAISS)**")
        st.markdown("""
        - Embeddings con Gemini (3072 dimensiones)
        - Búsqueda semántica con FAISS
        - Respuestas contextuales precisas
        - **Caché LRU** para acceso rápido
        - **Optimizado para free tier** (10 fragmentos/archivo)
        """)
    
    with col3:
        st.info("🤖 **Agentes IA con LangGraph**")
        st.markdown("""
        - **Explicar código**: Funciones, clases, algoritmos
        - **Revisar código**: Bugs, mejoras, optimizaciones
        - **Documentar**: Generación automática de documentación
        - **Router inteligente**: Clasifica y dirige consultas
        """)
    
    st.markdown("---")
    
    st.subheader("📚 Últimos repositorios analizados")
    
    repos = st.session_state.repo_service.list_repositories()
    
    if repos:
        for repo in repos[:5]:
            with st.container():
                cols = st.columns([3, 1, 1, 1, 1])
                
                with cols[0]:
                    st.write(f"**{repo['name']}**")
                    path = Path(repo['path'])
                    if path.exists():
                        st.caption(f"✅ {repo['path']}")
                    else:
                        st.caption(f"⚠️ {repo['path']} (no encontrado)")
                
                with cols[1]:
                    st.write(f"📁 {repo['file_count']} archivos")
                
                with cols[2]:
                    st.write(f"📊 {repo['total_lines']} líneas")
                
                with cols[3]:
                    created_at = repo['created_at']
                    if hasattr(created_at, 'strftime'):
                        fecha = created_at.strftime("%Y-%m-%d")
                    else:
                        fecha = str(created_at)[:10]
                    st.write(f"🕐 {fecha}")
                
                with cols[4]:
                   if st.button("Cargar", key=f"welcome_load_{repo['id']}"):
                        with st.spinner(f"Cargando repositorio {repo['name']}..."):
                            try:
                                # Reconstruir repositorio desde MySQL (sin regenerar embeddings)
                                repo_obj = st.session_state.repo_service.load_repository_from_db(repo['id'])
                                
                                if repo_obj:                                   
                                    
                                    # Crear RAGService (usará FAISS existente, no regenerará embeddings)
                                    rag_service = RAGService(
                                        repo_name=repo_obj.name,
                                        repo_path=repo_obj.path,
                                        repo_id=repo['id'],
                                        prefer_pro=st.session_state.prefer_pro,
                                        max_file_size_mb=st.session_state.max_file_size_mb,
                                        include_docs=st.session_state.include_docs
                                    )
                                    # NO llamar a index_repository() - usar vector store existente
                                    
                                    st.session_state.rag_service = rag_service
                                    st.session_state.current_repo = repo_obj
                                    st.session_state.repository_loaded = True
                                    st.session_state.messages = []
                                    
                                    st.session_state.agent_workflow = AgentWorkflow(rag_service)
                                    st.success(f"✅ Repositorio {repo['name']} cargado desde BD")
                                    st.success("🤖 Agentes LangGraph inicializados")
                                    st.rerun()
                                else:
                                    st.error("❌ Error cargando repositorio desde BD")
                                    
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                
                st.divider()
    else:
        st.info("👈 **Comienza cargando un repositorio** en la sección 'Cargar Repositorio'")
    
    with st.expander("📊 Información del Sistema", expanded=False):
        try:
            cache_stats = st.session_state.repo_service.cache.get_stats()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📦 Caché", f"{cache_stats['total_files']} archivos")
            with col2:
                st.metric("💾 Tamaño", f"{cache_stats['total_size_mb']:.1f} MB")
            with col3:
                st.metric("📊 Uso", f"{cache_stats['usage_percent']:.1f}%")
        except Exception as e:
            logger.debug(f"Error obteniendo estadísticas de caché: {e}")
        
        st.markdown("**🌐 Lenguajes soportados:**")
        languages = ["Python", "JavaScript", "TypeScript", "HTML", "CSS", "SCSS", "JSON", "SQL", "Shell", "Go", "Rust", "Java", "C++", "Ruby", "PHP"]
        st.markdown(" | ".join(languages[:8]))
        st.markdown(" | ".join(languages[8:]))
        
        st.markdown("**📐 Configuración de embeddings:**")
        st.markdown("- Modelo: Gemini Embedding-001")
        st.markdown("- Dimensión: 3072")
        st.markdown("- Fragmentos por archivo: 10")
        st.markdown("- Tamaño de fragmento: 500 caracteres")
        
        st.markdown("**🤖 Agentes disponibles:**")
        st.markdown("- **Router Agent**: Clasifica y dirige consultas automáticamente")
        st.markdown("- **Explain Agent**: Explica funciones y clases")
        st.markdown("- **Review Agent**: Revisa código y sugiere mejoras")
        st.markdown("- **Docs Agent**: Genera documentación automática")
        
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            st.success("✅ Gemini API: Configurada")
        else:
            st.warning("⚠️ Gemini API: No configurada")


def main() -> NoReturn:
    """
    Función principal que inicia la aplicación Streamlit.
    """
    logger.info("Iniciando AI Coding Assistant Local...")
    
    setup_project_structure()
    
    # Configuración de página (SOLO UNA VEZ)
    st.set_page_config(
        page_title="AI Coding Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # Barra lateral de navegación (SOLO AQUÍ)
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/robot.png", width=80)
        st.title("🤖 AI Coding Assistant")
        
        st.markdown("---")
        
        st.subheader("📋 Navegación")
        page = st.radio(
            "Ir a:",
            ["🏠 Inicio", "📤 Cargar Repositorio", "📊 Analizar", "💬 Chat", "📚 Repositorios", "⚙️ Configuración"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
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
            
            if 'agent_workflow' in st.session_state and st.session_state.agent_workflow:
                st.info("🤖 Agentes: Activos")
            else:
                st.warning("🤖 Agentes: No inicializados")
            
            if st.session_state.get('daily_limit_reached', False):
                st.warning("⚠️ Límite diario API alcanzado")
        else:
            st.warning("⏳ **Sin repositorio activo**")
        
        st.markdown("---")
        st.caption("v1.0.0 | Gemini + FAISS")
        st.caption("Multi-lenguaje | Agentes LangGraph")
        st.caption("Dimensión embeddings: 3072")
    
    # Mostrar página seleccionada (importaciones dinámicas)
    if page == "🏠 Inicio":
        show_welcome_page()
    elif page == "📤 Cargar Repositorio":
        from interface.streamlit.app import show_upload_section
        show_upload_section()
    elif page == "📊 Analizar":
        from interface.streamlit.app import show_analysis_section
        show_analysis_section()
    elif page == "💬 Chat":
        from interface.streamlit.app import show_chat_section
        show_chat_section()
    elif page == "📚 Repositorios":
        from interface.streamlit.app import show_repositories_list
        show_repositories_list()
    else:
        from interface.streamlit.app import show_configuration_section
        show_configuration_section()
    
    logger.debug("Página renderizada correctamente")


if __name__ == "__main__":
    main()