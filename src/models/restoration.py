import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from typing import Optional
from contextlib import nullcontext

class ImageRestorationPipeline:
    """
    Clase encargada de encapsular el pipeline de Hugging Face Diffusers
    para tareas de restauración de imágenes (denoising / deblurring / image-to-image)
    combinando preprocesamiento clásico y difusión latente.
    """
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model_id = model_id
        self.pipeline: Optional[StableDiffusionImg2ImgPipeline] = None

    def load_model(self) -> None:
        """Carga el modelo en memoria utilizando el dispositivo y tipo de dato óptimos."""
        if self.pipeline is not None:
            return
        
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            low_cpu_mem_usage=True
        )
        self.pipeline.to(self.device)
        
        # Optimización para memoria si estamos en CUDA
        if self.device == "cuda":
            self.pipeline.enable_attention_slicing()

    def _preprocess_denoise(self, image: Image.Image) -> Image.Image:
        """
        Aplica un filtro Non-Local Means de OpenCV para remover ruido fotográfico grueso
        preservando los bordes y la estructura geométrica antes de la difusión.
        """
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Aplicar Non-Local Means Denoising para color
        denoised_bgr = cv2.fastNlMeansDenoisingColored(
            img_bgr, 
            None, 
            h=10,          # Intensidad del filtro para luminosidad
            hColor=10,     # Intensidad del filtro para color
            templateWindowSize=7, 
            searchWindowSize=21
        )
        
        return Image.fromarray(cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB))

    def restore_image(
        self, 
        image: Image.Image, 
        prompt: str = "High quality, sharp focus, detailed, restored photograph, clear edges",
        negative_prompt: str = "cartoon, painting, illustration, distorted faces, blurry, artifacts, deformed, low resolution",
        strength: float = 0.20,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 40
    ) -> Image.Image:
        """
        Procesa una imagen degradada combinando preprocesamiento clásico y modelo img2img.
        
        Args:
            image: Imagen PIL degradada.
            prompt: Descripción orientativa para guiar la restauración semántica.
            negative_prompt: Elementos a evitar en la generación.
            strength: Controla el nivel de alteración (se recomienda 0.15 - 0.25 para ruido).
            guidance_scale: Escala de guía para el prompt.
            num_inference_steps: Pasos de difusión (aumentar a 40 para mayor refinamiento).
        
        Returns:
            Imagen PIL restaurada.
        """
        if self.pipeline is None:
            self.load_model()

        # 1. Preprocesamiento: Limpieza de ruido base con visión clásica
        cleaned_image = self._preprocess_denoise(image)

        # 2. Asegurar dimensiones compatibles con Stable Diffusion (múltiplos de 8)
        base_width, base_height = cleaned_image.size
        new_width = (base_width // 8) * 8
        new_height = (base_height // 8) * 8
        
        processed_image = cleaned_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 3. Ejecución de la inferencia por difusión
        context = torch.autocast(self.device) if self.device == "cuda" else nullcontext()
        
        with context:
            result = self.pipeline(
                prompt=prompt,
                image=processed_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_prompt
            ).images[0]

        # Devolver a las dimensiones originales
        return result.resize((base_width, base_height), Image.Resampling.LANCZOS)