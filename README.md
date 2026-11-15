# 🛡️ Diffusion Restore & Inspect

> **Trabajo Integrador:** MVP funcional de restauración de imágenes (*Denoising / Deblurring*) mediante modelos de difusión, inspección visual asistida por Inteligencia Artificial y controles de transparencia ética (HAI - Human-AI Interaction).

---

## 📋 Descripción del Proyecto

**Diffusion Restore & Inspect** es una aplicación modular desarrollada en Python para abordar la tarea de **Denoising y Deblurring** (Opción B) en la restauración de imágenes degradadas. El sistema combina el poder generativo de los modelos de difusión estocásticos con modelos de lenguaje visual (VLM) para auditar semánticamente los cambios y un panel de transparencia basado en mapas de diferencias absolutas. Esto permite distinguir de forma clara y auditable qué información fue real y qué porción fue rellenada o alterada por el *prior* del modelo.

---

## 🛠️ Stack Tecnológico

El proyecto está diseñado bajo los principios de *Clean Code*, arquitectura modular y gestión moderna de dependencias:
* **Lenguaje:** Python 3.12
* **Gestor de Entorno y Paquetes:** `uv` (herramienta de alto rendimiento para flujos limpios y reproducibles).
* **Interfaz de Usuario:** Streamlit (despliegue web local interactivo).
* **Deep Learning & IA:** 
  * PyTorch (`torch`) para el soporte de tensores y ejecución en GPU/CPU.
  * Hugging Face `diffusers` (pipeline de *Image-to-Image* / *Stable Diffusion*).
  * Hugging Face `transformers` (BLIP para subtitulado y comprensión semántica automatizada).
* **Procesamiento de Imágenes y Métricas:** Pillow (PIL), NumPy, OpenCV y Matplotlib.

---

## 🧠 Modelos de IA Utilizados

El sistema integra dos componentes principales basados en modelos preentrenados descargados de **Hugging Face Hub**:

1. **Restauración (Image-to-Image / Denoising):**
   * **Modelo:** `runwayml/stable-diffusion-v1-5` (configurado mediante `StableDiffusionImg2ImgPipeline`).
   * **Naturaleza:** Modelo preentrenado de difusión latente, adaptado para tareas de *img2img*.
   * **Funcionamiento:** Toma una imagen degradada (ruido o desenfoque), la proyecta al espacio latente y aplica un proceso de difusión guiado por un prompt semántico y un nivel de alteración controlado (`strength`), permitiendo limpiar ruido manteniendo la estructura subyacente.
   * **Optimizaciones:** Ejecución opcional en precisión mixta (`float16` en CUDA) y segmentación de atención (`enable_attention_slicing`) para optimizar el consumo de memoria VRAM/RAM.

2. **Inspección Visual (VLM - Vision Language Model):**
   * **Modelo:** `Salesforce/blip-image-captioning-base` (`BlipForConditionalGeneration` y `BlipProcessor`).
   * **Naturaleza:** Modelo preentrenado de subtitulado de imágenes.
   * **Funcionamiento:** Analiza de manera independiente tanto la imagen original como la imagen restaurada para generar descripciones automáticas en lenguaje natural, permitiendo auditar si la semántica clave del contenido se ha conservado o alterado.

---

## 📂 Estructura del Proyecto

```text
diffusion-restore-inspect/
├── src/
│   ├── models/
│   │   ├── restoration.py    # Pipeline de Hugging Face Diffusers (img2img)
│   │   └── inspection.py     # Pipeline de subtitulado con BLIP VLM
│   ├── utils/
│   │   └── image_ops.py      # Operaciones de imagen y mapas de diferencias (HAI)
│   └── app.py                # Interfaz web principal en Streamlit
├── pyproject.toml            # Configuración de dependencias y empaquetado con uv
└── README.md                 # Documentación técnica del proyecto
```

---

## ⚙️ Configuración y Despliegue Local

Para levantar el proyecto en tu entorno local utilizando `uv`, sigue estos pasos desde la terminal:

1. **Clonar o situarse en la raíz del repositorio:**
   ```powershell
   cd C:\repos\diffusion-restore-inspect
   ```

2. **Sincronizar el entorno y las dependencias con `uv`:**
   El gestor leerá automáticamente el archivo `pyproject.toml` y configurará el entorno virtual `.venv`:
   ```powershell
   uv sync
   uv pip install -e .
   ```
   *(Opcional recomendado para acelerar la carga de pesos y optimizar el procesamiento de imágenes)*:
   ```powershell
   uv pip install accelerate torchvision
   ```

3. **Ejecutar la aplicación web local con Streamlit:**
   Invoca el comando a través de `uv run`:
   ```powershell
   uv run streamlit run src/app.py
   ```

4. **Acceso:**
   Abre el navegador web en la URL local generada (por defecto: `http://localhost:8501`).

---

## 🖥️ Guía de Utilización de la Interfaz

Una vez abierta la aplicación en Streamlit, encontrarás el siguiente flujo de trabajo:

1. **Barra Lateral (Configuración):**
   * **Carga de Archivos (`File Uploader`):** Sube tu imagen degradada en formato JPG, JPEG o PNG.
   * **Parámetros de Difusión:**
     * *Fuerza de alteración (`Strength`):* Controla cuánto interviene el modelo. Para fotografías con ruido denso, se recomienda usar valores bajos (**`0.15` a `0.25`**) para evitar alucinaciones o distorsión de rostros.
     * *Prompt de Guía Semántica:* Texto descriptivo positivo para orientar la restauración (ej: *"Minimal denoising, sharp photograph, clear details"*).
     * *Prompt Negativo:* Elementos que el modelo debe evitar activamente (ej: *"cartoon, painting, blurry, distorted faces, artifacts"*).
   * **Botón de Ejecución:** Haz clic en *"🚀 Ejecutar Restauración e Inspección"* para iniciar el pipeline.

2. **Panel de Resultados y Auditoría Visual:**
   * **Imagen Original (Degradada):** Muestra el insumo inicial junto con su descripción automática generada por el VLM.
   * **Imagen Restaurada (Difusión):** Muestra el resultado limpio obtenido por el modelo de difusión, acompañado de su respectivo *caption* semántico.
   * **Mapa de Diferencias (HAI):** Muestra una visualización diferencial absoluta entre la imagen original y la restaurada. Las zonas más brillantes resaltan los píxeles donde el modelo intervino o rellenó información.

3. **Panel de Transparencia (HAI - Honest AI):**
   * Una sección informativa dedicada a recordar la naturaleza estocástica del modelo y la importancia de contrastar el mapa de diferencias y las descripciones del VLM para asegurar que la semántica original no haya sido alterada artificialmente.

---

## 📜 Declaración Responsable y Ética

Este MVP incorpora principios de **Inteligencia Artificial Honesta (HAI)** al no ocultar las modificaciones realizadas por los modelos generativos. El mapa de diferencias actúa como una herramienta de auditoría técnica indispensable para entornos donde la fidelidad visual y probatoria de una fotografía es crítica.