import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from typing import Optional
from contextlib import nullcontext

class ImageRestorationPipeline:
    """
    Clase encargada de encapsular el pipeline de Hugging Face Diffusers
    para tareas de restauración de imágenes (denoising / deblurring / image-to-image).
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

        # Nota: Para tareas estrictas de Denoising/Deblurring puros se pueden usar pipelines 
        # especializados de Image-to-Image o Stable Diffusion con un strength ajustado.
        from diffusers import StableDiffusionImg2ImgPipeline
        
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

    def restore_image(
        self, 
        image: Image.Image, 
        prompt: str = "High quality, sharp focus, detailed, restored photograph, clear edges",
        negative_prompt: str = "blurry, noisy, grainy, distorted, low resolution, artifacts",
        strength: float = 0.35,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30
    ) -> Image.Image:
        """
        Procesa una imagen degradada (ruido/desenfoque) aplicando el modelo de difusión img2img.
        
        Args:
            image: Imagen PIL degradada.
            prompt: Descripción orientativa para guiar la restauración semántica.
            negative_prompt: Elementos a evitar en la generación.
            strength: Controla el nivel de alteración (valores bajos como 0.3-0.4 preservan la estructura original).
            guidance_scale: Escala de guía para el prompt.
            num_inference_steps: Pasos de difusión.
        
        Returns:
            Imagen PIL restaurada.
        """
        if self.pipeline is None:
            self.load_model()

        # Asegurar dimensiones compatibles con Stable Diffusion (múltiplos de 8)
        base_width, base_height = image.size
        new_width = (base_width // 8) * 8
        new_height = (base_height // 8) * 8
        
        processed_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Ejecución de la inferencia utilizando contextlib.nullcontext
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

        # Devolver a las dimensiones originales si es necesario
        return result.resize((base_width, base_height), Image.Resampling.LANCZOS)