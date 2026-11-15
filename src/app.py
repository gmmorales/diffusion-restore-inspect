import sys
from pathlib import Path

# Añadir la raíz del proyecto al sys.path para garantizar importaciones limpias de src.*
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from PIL import Image
import numpy as np

from src.models.restoration import ImageRestorationPipeline
from src.models.inspection import VisualInspectionPipeline
from src.utils.image_ops import compute_difference_map

# Configuración de la página
st.set_page_config(
    page_title="Diffusion Restore & Inspect (MVP)",
    page_icon="🔍",
    layout="wide"
)

# Caché para la carga de modelos para optimizar recursos
@st.cache_resource
def load_restoration_model():
    pipeline = ImageRestorationPipeline()
    pipeline.load_model()
    return pipeline

@st.cache_resource
def load_inspection_model():
    inspector = VisualInspectionPipeline()
    inspector.load_model()
    return inspector

def main():
    st.title("🛡️ Diffusion Restore & Inspect")
    st.markdown(
        """
        **Trabajo Integrador 2:** MVP funcional de restauración de imágenes (*Denoising / Deblurring*) 
        mediante modelos de difusión, inspección visual asistida por IA y controles de transparencia (HAI).
        """
    )

    # Sidebar para controles y parámetros
    st.sidebar.header("⚙️ Configuración del Pipeline")
    
    upload_file = st.sidebar.file_uploader(
        "Sube una imagen degradada (ruido / blur)", 
        type=["jpg", "jpeg", "png"]
    )
    
    st.sidebar.subheader("Parámetros de Difusión")
    strength = st.sidebar.slider(
        "Fuerza de alteración (Strength)", 
        min_value=0.1, max_value=0.8, value=0.20, step=0.05,
        help="Usa valores bajos (0.15 - 0.25) para fotos con ruido para evitar que el modelo alucine o distorsione la imagen."
    )
    
    prompt = st.sidebar.text_input(
        "Prompt de guía semántica", 
        value="Minimal denoising, sharp photograph, clear details, authentic texture, no painting"
    )

    negative_prompt = st.sidebar.text_input(
        "Prompt Negativo", 
        value="cartoon, painting, illustration, distorted faces, blurry, artifacts, deformed, low resolution"
    )

    if upload_file is not None:
        original_image = Image.open(upload_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📥 Imagen Original (Degradada)")
            st.image(original_image, width='stretch')

        # Carga de modelos
        with st.spinner("Cargando modelos de Hugging Face en memoria..."):
            restorer = load_restoration_model()
            inspector = load_inspection_model()

        # Botón de ejecución
        if st.sidebar.button("🚀 Ejecutar Restauración e Inspección", type="primary"):
            with st.spinner("Procesando imagen con modelo de difusión y analizando con VLM..."):
                # 1. Restauración
                restored_image = restorer.restore_image(
                    image=original_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    strength=strength
                )
                
                # 2. Inspección Visual (VLM Captions)
                caption_original = inspector.generate_caption(original_image, prompt="a photo of")
                caption_restored = inspector.generate_caption(restored_image, prompt="a photo of")
                
                # 3. Mapa de Diferencias (HAI)
                diff_map = compute_difference_map(original_image, restored_image)

            # Resultados visuales
            st.markdown("---")
            st.header("📊 Resultados del Pipeline y Auditoría Visual")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.subheader("Original")
                st.image(original_image, width='stretch')
                st.caption(f"**VLM Caption:** *{caption_original}*")
                
            with res_col2:
                st.subheader("Restaurada (Difusión)")
                st.image(restored_image, width='stretch')
                st.caption(f"**VLM Caption:** *{caption_restored}*")

            with res_col3:
                st.subheader("🔍 Mapa de Diferencias (HAI)")
                st.image(diff_map, width='stretch')
                st.caption("Las zonas más brillantes indican mayor intervención / modificación del modelo.")

            # Panel de Transparencia y Honestidad (HAI)
            st.markdown("---")
            st.info(
                "💡 **Nota de Transparencia (HAI):** Este sistema utiliza un modelo de difusión estocástico. "
                "El mapa de diferencias resalta las regiones donde el *prior* del modelo ha rellenado o alterado información. "
                "Verifique las descripciones automáticas del VLM para asegurar que la semántica clave no haya sido alterada artificialmente."
            )
    else:
        st.info("👈 Por favor, sube una imagen en la barra lateral para comenzar con la restauración.")

if __name__ == "__main__":
    main()