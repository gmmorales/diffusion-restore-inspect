import numpy as np
from PIL import Image, ImageChops

def compute_difference_map(original_img: Image.Image, restored_img: Image.Image) -> Image.Image:
    """
    Calcula el mapa de diferencias absolutas entre la imagen degradada y la restaurada.
    Permite visualizar visualmente dónde el modelo de difusión intervino o alucinó píxeles.
    """
    # Asegurar que ambas imágenes tengan el mismo tamaño y modo
    restored_img = restored_img.resize(original_img.size)
    
    # Obtener diferencia absoluta
    diff = ImageChops.difference(original_img.convert("RGB"), restored_img.convert("RGB"))
    
    # Realzar el contraste del mapa de diferencias para mejor visualización
    diff_np = np.array(diff)
    diff_np = np.clip(diff_np * 3.0, 0, 255).astype(np.uint8)
    
    return Image.fromarray(diff_np)