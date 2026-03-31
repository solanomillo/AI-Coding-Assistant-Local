# 🤖 AI Coding Assistant Local

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.32.0-red.svg)
![LangGraph](https://img.shields.io/badge/langgraph-0.0.40-green.svg)
![Gemini](https://img.shields.io/badge/gemini-2.5--flash-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Un asistente de IA local para analizar repositorios de código usando RAG y agentes especializados**

[Características](#-características) •
[Instalación](#-instalación) •
[Uso](#-uso) •
[Arquitectura](#-arquitectura) •
[Contribuir](#-contribuir)

</div>

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Tecnologías](#-tecnologías)
- [Instalación](#-instalación)
- [Configuración](#-configuración)
- [Uso](#-uso)
- [Arquitectura](#-arquitectura)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Optimizaciones](#-optimizaciones)
- [Pruebas](#-pruebas)
- [Roadmap](#-roadmap)
- [Contribuciones](#-contribuciones)
- [Licencia](#-licencia)

---

## 🎯 Descripción

**AI Coding Assistant Local** es una aplicación que permite analizar repositorios de código de manera inteligente utilizando técnicas de **RAG (Retrieval Augmented Generation)** y **agentes especializados** orquestados con LangGraph. El sistema indexa el código fuente, genera embeddings con Gemini, y permite realizar preguntas en lenguaje natural sobre el repositorio.

### ¿Qué hace?

- 📁 **Carga repositorios** desde archivos ZIP o directorios locales
- 🔍 **Analiza el código** extrayendo funciones, clases, imports y estructura
- 🧠 **Genera embeddings** con Gemini para búsqueda semántica
- 💾 **Almacena vectores** en FAISS para recuperación eficiente
- 🤖 **Responde preguntas** usando agentes especializados (explicar, revisar, documentar)
- 📊 **Genera documentación** automática del código

---

## ✨ Características

### 📁 Análisis de Repositorios
- Soporte para **múltiples lenguajes**: Python, JavaScript, TypeScript, HTML, CSS
- Extracción de **funciones, clases, imports y dependencias**
- Detección automática de **proyectos Django**
- Filtros de seguridad para ignorar archivos sensibles (`.env`, credenciales)

### 🔍 Sistema RAG (Gemini + FAISS)
- **Embeddings** con Gemini (3072 dimensiones)
- **Búsqueda semántica** con FAISS
- **Caché LRU** para fragmentos de código
- Procesamiento por lotes para optimizar consumo de API

### 🤖 Agentes con LangGraph
| Agente | Función |
|--------|---------|
| **Router** | Clasifica consultas y dirige al agente adecuado |
| **Explain** | Explica funciones, clases y bloques de código |
| **Review** | Revisa código, sugiere mejoras y detecta bugs |
| **Docs** | Genera documentación automática |

### ⚡ Optimizaciones
- **Límite de fragmentos** (10 por archivo) para ahorrar cuota
- **Caché de estado de API** (5 minutos si OK, 1 hora si agotada)
- **Procesamiento en lotes** de 20 fragmentos
- **Manejo amigable** de errores de cuota

---

## 🛠️ Tecnologías

| Categoría | Tecnologías |
|-----------|-------------|
| **Frontend** | Streamlit |
| **LLM** | Google Gemini (2.5 Flash / 2.5 Pro) |
| **Embeddings** | Gemini Embedding-001 (3072 dim) |
| **Vector DB** | FAISS |
| **Orquestación** | LangGraph |
| **Base de Datos** | MySQL |
| **Parsers** | AST (Python), Regex (JS, HTML, CSS) |
| **Lenguajes** | Python 3.12+ |

---

## 📦 Instalación

### Requisitos Previos

- Python 3.12 o superior
- MySQL 8.0+
- Git

### Pasos de Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/solanomillo/AI-Coding-Assistant-Local.git
cd ai-coding-assistant

# 2. Crear entorno virtual
python -m venv venv

# Activar en Windows
venv\Scripts\activate
# Activar en Linux/Mac
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar base de datos MySQL
mysql -u root -p
CREATE DATABASE ai_coding_assistant;
exit;

# 5. Ejecutar migraciones (opcional)
python scripts/init_database.py

# 6. Configurar variables de entorno
cp .env.example .env
# Editar .env con tu API key de Gemini y credenciales de MySQL

# 7. Ejecutar la aplicación
streamlit run main.py
```

---

## ⚙️ Configuración
### Variables de Entorno (.env)
```bash
# Gemini API
GEMINI_API_KEY=tu_api_key_aqui
GEMINI_PREFER_PRO=false

# MySQL Database
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=tu_contraseña
DB_NAME=ai_coding_assistant
DB_PORT = 

# Application
LOG_LEVEL=INFO
MAX_CACHE_SIZE_MB=500
MAX_FILE_SIZE_MB=1
MAX_FRAGMENTS_PER_FILE=10
```

---

##Obtener API Key de Gemini
###Ve a Google AI Studio

1.**Inicia sesión con tu cuenta Google

2.**Crea una nueva API key

3.**Copia la clave y pégala en .env o ingresala por la interfaz grafca y prueba su conexión.

---

## 🚀 Uso

### 1. Iniciar la Aplicación

```bash
streamlit run main.py
```

### 2. Cargar un Repositorio

- **Archivo ZIP:** Selecciona un archivo ZIP con el código fuente
- **Directorio local:** Ingresa la ruta absoluta del directorio

### 3. Analizar el Código

Una vez indexado, puedes hacer preguntas como:

| Tipo | Ejemplo |
|------|---------|
| General | `"¿De qué trata este repositorio?"` |
| Explicar | `"Explica la función calcular_total"` |
| Revisar | `"Revisa el archivo utils.py por errores"` |
| Documentar | `"Genera documentación para la clase Usuario"` |
| Estadísticas | `"¿Cuántos archivos tiene el repositorio?"` |

---

### 4. Seleccionar Modelo

Puedes elegir entre:

- ⚡ **Flash:** Gratuito, rápido (recomendado)
- ⭐ **Pro:** Mayor capacidad (puede tener costo)

### 5. Ver Resultados

- Respuestas con contexto preciso
- Fuentes consultadas (archivos utilizados)
- Estadísticas de procesamiento

---

## 🏗️ Arquitectura
```text
┌─────────────────────────────────────────────────────────────────┐
│                     INTERFAZ (Streamlit)                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │   Cargar    │ │  Analizar   │ │    Chat     │              │
│  │ Repositorio │ │  Código     │ │  con IA     │              │
│  └─────────────┘ └─────────────┘ └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   APLICACIÓN (Agentes + RAG)                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   LangGraph Workflow                     │   │
│  │  Router → Explain/Review/Docs → Respuesta              │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     RAG Service                         │   │
│  │  Embeddings → FAISS → Recuperación → LLM               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   INFRAESTRUCTURA                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│  │ Gemini   │ │  FAISS   │ │  MySQL   │ │  Caché   │         │
│  │ LLM/Embed│ │  Vector  │ │ Metadatos│ │   LRU    │         │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

##📁 Estructura del Proyecto
```text
ai-coding-assistant/
├── main.py                          # Punto de entrada
├── requirements.txt                 # Dependencias
├── .env                             # Variables de entorno
├── .gitignore                       # Archivos ignorados
│
├── interface/
│   └── streamlit/
│       └── app.py                   # UI completa
│
├── application/
│   ├── agents/                      # Agentes LangGraph
│   │   ├── base_agent.py
│   │   ├── router_agent.py
│   │   ├── explain_agent.py
│   │   ├── review_agent.py
│   │   └── docs_agent.py
│   ├── services/                    # Servicios
│   │   ├── repo_service.py
│   │   ├── rag_gemini_service.py
│   │   ├── cache_service.py
│   │   └── service_factory.py
│   └── graph/
│       └── workflow.py              # Orquestación LangGraph
│
├── domain/
│   └── models/
│       └── repository.py            # Modelos de datos
│
├── infrastructure/
│   ├── embeddings/
│   │   └── gemini_embedding.py
│   ├── vector_db/
│   │   └── faiss_store.py
│   ├── llm_clients/
│   │   ├── gemini_llm.py
│   │   └── error_handler.py
│   ├── database/
│   │   └── mysql_repository.py
│   └── parsers/                     # Multi-lenguaje
│       ├── base_parser.py
│       ├── python_parser.py
│       ├── javascript_parser.py
│       ├── html_parser.py
│       └── css_parser.py
│
└── data/                            # Datos persistentes
    ├── repositories/                # Copias de repositorios
    ├── vectors/                     # Índices FAISS
    └── cache/                       # Caché LRU
```

---

## ⚡ Optimizaciones

| Optimización | Descripción | Beneficio |
|--------------|-------------|-----------|
| Caché de API | Estado de API cacheado 5-60 min | -80% llamadas |
| Límite de fragmentos | 10 fragmentos por archivo | -70% consumo |
| Procesamiento por lotes | 20 fragmentos por lote | Control de rate limit |
| Filtros de archivos | Ignora .env, node_modules, etc. | Seguridad + rendimiento |
| Caché LRU | Fragmentos recientes en disco | Acceso rápido |
| Modelos configurables | Usuario elige Flash/Pro | Control de costos |

---

##🧪 Pruebas
### Script de prueba de conexión
```bash
python -c "
from infrastructure.llm_clients.gemini_llm import GeminiLLM
llm = GeminiLLM()
print(f'Modelo: {llm.current_model}')
print(f'Disponible: OK')
"
```
## Verificar modelos disponibles
```bash
python scripts/list_models.py
```
## Limpiar datos
```bash
# Limpiar base de datos
mysql -u root -p -e "USE ai_coding_assistant; DELETE FROM repositories;"

# Limpiar vectores y caché
rm -rf data/vectors/* data/cache/files/* data/repositories/*
```

## 🗺️ Roadmap

### Versión 1.0 (Actual)
- ✅ Carga de repositorios (ZIP y directorio)
- ✅ Análisis multi-lenguaje
- ✅ RAG con Gemini + FAISS
- ✅ Agentes LangGraph
- ✅ Interfaz Streamlit
- ✅ Gestión de cuota

### Versión 2.0 (Futuro cercano)
- ⬜ Soporte para más lenguajes (Go, Rust, Java)
- ⬜ Análisis de dependencias
- ⬜ Detección de vulnerabilidades
- ⬜ Exportación de reportes (PDF, Markdown)
- ⬜ Integración con GitHub API

### Versión 3.0 (Futuro lejano)
- ⬜ Despliegue en la nube (Docker, Kubernetes)
- ⬜ Autenticación de usuarios
- ⬜ Colaboración en tiempo real
- ⬜ Plugins personalizables

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, sigue estos pasos:

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'feat: agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

### Convenciones de código

- **PEP8** para Python
- **Type hints** en todas las funciones
- **Docstrings** en español
- **Importaciones** al inicio del archivo
- **Logs** sin emojis (solo UI)

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Ver el archivo `LICENSE` para más detalles.

---

## 🙏 Agradecimientos

- **Google Gemini** por la API
- **LangChain** por LangGraph
- **FAISS** por la búsqueda vectorial
- **Streamlit** por la interfaz
