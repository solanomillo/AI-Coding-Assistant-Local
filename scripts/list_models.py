#!/usr/bin/env python
"""
Script para listar todos los modelos disponibles de Gemini API.
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

def list_available_models():
    """Lista todos los modelos disponibles con sus capacidades."""
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY no configurada en .env")
        return
    
    print("=" * 60)
    print("CONSULTANDO MODELOS DISPONIBLES DE GEMINI API")
    print("=" * 60)
    
    try:
        genai.configure(api_key=api_key)
        
        # Listar todos los modelos
        models = genai.list_models()
        
        print("\n📋 MODELOS DISPONIBLES:\n")
        
        for model in models:
            # Filtrar solo modelos de generación de texto
            if 'generateContent' in model.supported_generation_methods:
                print(f"✅ {model.name}")
                print(f"   - Display name: {model.display_name}")
                print(f"   - Description: {model.description[:100]}..." if model.description else "   - Description: N/A")
                print(f"   - Supported methods: {model.supported_generation_methods}")
                print()
        
        print("=" * 60)
        print("RECOMENDACIONES:")
        print("=" * 60)
        print("🔹 Para FREE TIER (gratuito):")
        print("   - gemini-1.5-flash (recomendado)")
        print("   - gemini-1.5-flash-8b (más pequeño)")
        print("   - gemini-2.0-flash (si está disponible)")
        print()
        print("🔹 Para PRO (pago):")
        print("   - gemini-1.5-pro")
        print("   - gemini-2.0-pro-exp-02-05 (experimental)")
        print()
        print("🔹 Embeddings:")
        print("   - models/embedding-001 (3072 dimensiones)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    list_available_models()