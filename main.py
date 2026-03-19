"""
Módulo principal del proyecto AI Coding Assistant Local.

Este archivo sirve como punto de entrada para la aplicación,
configurando el logging y la ejecución inicial del sistema.
"""

import logging
import sys
from pathlib import Path
from typing import NoReturn
import streamlit as st
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ai_coding_assistant.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


def setup_project_structure() -> None:
    """Verifica y crea la estructura de directorios necesaria."""
    logger.info("Verificando estructura de directorios...")
    
    required_dirs = [
        "data/repositories",
        "data/repositories/temp",
        "data/vector_store",  # Añadido para FAISS
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


def initialize_session_state() -> None:
    """Inicializa las variables de estado de la sesión."""
    if 'repository_loaded' not in st.session_state:
        st.session_state.repository_loaded = False
    
    if 'current_repo' not in st.session_state:
        st.session_state.current_repo = None
    
    if 'repo_service' not in st.session_state:
        from application.services.repo_service import RepositoryService
        st.session_state.repo_service = RepositoryService()
        logger.info("Servicio de repositorios inicializado")


def show_welcome_page() -> None:
    """Muestra la página de bienvenida."""
    st.title("🤖 AI Coding Assistant Local")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("📁 **Analiza repositorios**")
        st.markdown("""
        - Carga repositorios vía ZIP o directorio local
        - Analiza estructura de código automáticamente
        - Extrae funciones, clases y metadatos
        """)
    
    with col2:
        st.info("🔍 **RAG System (Gemini + FAISS)**")
        st.markdown("""
        - Embeddings con Gemini (768 dimensiones)
        - Búsqueda semántica con FAISS
        - Respuestas contextuales precisas
        """)
    
    with col3:
        st.info("🤖 **Agentes IA**")
        st.markdown("""
        - Explicación de código
        - Code review automático
        - Generación de documentación
        """)
    
    st.markdown("---")
    
    # Mostrar últimos repositorios
    st.subheader("📚 Últimos repositorios analizados")
    repos = st.session_state.repo_service.list_repositories()
    
    if repos:
        for repo in repos[:5]:
            with st.container():
                cols = st.columns([3, 1, 1, 1])
                cols[0].write(f"**{repo['name']}**")
                cols[1].write(f"📁 {repo['file_count']} archivos")
                cols[2].write(f"📊 {repo['total_lines']} líneas")
                
                # 🔥 CORRECCIÓN: Manejar datetime correctamente
                created_at = repo['created_at']
                if isinstance(created_at, datetime):
                    fecha_str = created_at.strftime("%Y-%m-%d")
                else:
                    # Si es string, tomar primeros 10 caracteres
                    fecha_str = str(created_at)[:10] if created_at else "Fecha desconocida"
                
                cols[3].write(f"🕐 {fecha_str}")
    else:
        st.info("👈 Comienza cargando un repositorio en la sección 'Cargar Repositorio'")


def main() -> NoReturn:
    """Función principal."""
    logger.info("Iniciando AI Coding Assistant Local...")
    
    # Verificar estructura
    setup_project_structure()
    
    # Configuración de página
    st.set_page_config(
        page_title="AI Coding Assistant Local",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inicializar estado
    initialize_session_state()
    
    # Barra lateral
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/robot.png", width=80)
        st.title("🤖 AI Coding Assistant")
        
        st.divider()
        
        # Navegación
        st.subheader("📋 Navegación")
        page = st.radio(
            "Ir a:",
            ["Inicio", "Cargar Repositorio", "Analizar", "Chat", "Repositorios", "Configuración"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Estado del sistema
        st.subheader("📊 Estado")
        if st.session_state.repository_loaded and st.session_state.current_repo:
            st.success(f"✅ Repositorio activo:\n{st.session_state.current_repo.name}")
            
            repo = st.session_state.current_repo
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Archivos", repo.file_count)
            with col2:
                st.metric("Líneas", repo.total_lines)
            
            # Mostrar si está indexado
            if 'rag_service' in st.session_state:
                st.info("🧠 Indexado con Gemini")
        else:
            st.warning("⏳ Sin repositorio activo")
        
        st.divider()
        st.caption("v1.0.0 | Gemini + FAISS")
    
    # Cargar página
    if page == "Inicio":
        show_welcome_page()
    
    elif page == "Cargar Repositorio":
        from interface.streamlit.app import show_upload_section
        show_upload_section()
    
    elif page == "Analizar":
        from interface.streamlit.app import show_analysis_section
        show_analysis_section()
    
    elif page == "Chat":
        from interface.streamlit.app import show_chat_section
        show_chat_section()
    
    elif page == "Repositorios":
        from interface.streamlit.app import show_repositories_list
        show_repositories_list()
    
    else:  # Configuración
        from interface.streamlit.app import show_configuration_section
        show_configuration_section()


if __name__ == "__main__":
    main()