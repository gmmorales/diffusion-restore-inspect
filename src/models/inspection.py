import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Optional

class VisualInspectionPipeline:
    """
    Clase para realizar análisis visual e inspección de las imágenes 
    utilizando Vision-Language Models (VLM) de Hugging Face (ej. BLIP).
    """
    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-base", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.processor: Optional[BlipProcessor] = None
        self.model: Optional[BlipForConditionalGeneration] = None

    def load_model(self) -> None:
        """Carga el procesador y el modelo VLM en memoria."""
        if self.model is not None:
            return

        self.processor = BlipProcessor.from_pretrained(self.model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_id)
        self.model.to(self.device)

    def generate_caption(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """
        Genera una descripción textual de la imagen para evaluar la comprensión 
        semántica y verificar si los detalles clave se preservaron tras la restauración.
        """
        if self.model is None:
            self.load_model()

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=50)

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption