"""
Módulo principal del proyecto AI Coding Assistant Local.

Este archivo sirve como punto de entrada para la aplicación,
configurando el logging y la ejecución inicial del sistema.

Example:
    Para ejecutar la aplicación:
        $ streamlit run main.py
"""

import logging
import sys
from pathlib import Path
from typing import NoReturn
import streamlit as st

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ai_coding_assistant.log')
    ]
)

logger = logging.getLogger(__name__)


def setup_project_structure() -> None:
    """
    Verifica y crea la estructura de directorios necesaria para el proyecto.
    
    Esta función asegura que todos los directorios requeridos existan
    antes de iniciar la aplicación.
    """
    logger.info("Verificando estructura de directorios...")
    
    required_dirs = [
        "data/repositories",
        "interface/streamlit",
        "application/agents",
        "application/services",
        "application/graph",
        "domain/models",
        "infrastructure/embeddings",
        "infrastructure/vector_db",
        "infrastructure/llm_clients"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directorio creado: {dir_path}")
        else:
            logger.debug(f"Directorio existente: {dir_path}")


def main() -> NoReturn:
    """
    Función principal que inicia la aplicación Streamlit.
    
    Returns:
        NoReturn: Esta función no retorna, mantiene la aplicación en ejecución.
    """
    logger.info("Iniciando AI Coding Assistant Local...")
    
    # Verificar estructura del proyecto
    setup_project_structure()
    

    
    
    # Configuración de página
    st.set_page_config(
        page_title="AI Coding Assistant Local",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Mostrar pantalla de bienvenida
    st.title("🤖 AI Coding Assistant Local")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("📁 **Analiza repositorios**")
        st.markdown("Carga y analiza código fuente de diferentes lenguajes")
    
    with col2:
        st.info("🔍 **RAG System**")
        st.markdown("Búsqueda semántica en tu base de código")
    
    with col3:
        st.info("🤖 **Agentes IA**")
        st.markdown("Agentes especializados en análisis de código")
    
    st.markdown("---")
    st.warning("👈 Utiliza el menú lateral para navegar entre las funcionalidades")
    
    logger.info("Aplicación iniciada correctamente")


if __name__ == "__main__":
    main()