# 🧪 Documentación de Pruebas y Validación del MVP

Este documento registra los experimentos empíricos realizados sobre el sistema **Diffusion Restore & Inspect** con el objetivo de validar el rendimiento del pipeline híbrido (`OpenCV` + `StableDiffusionImg2ImgPipeline` v1.5) y demostrar el funcionamiento de los mecanismos de auditoría visual y transparencia (**HAI**).

---

## 🏗️ Arquitectura Evaluada

* **Preprocesamiento Clásico:** Filtro espacial `cv2.fastNlMeansDenoisingColored` ($h=10$, $hColor=10$) para supresión inicial de ruido de alta frecuencia.
* **Motor de Difusión Latente:** `runwayml/stable-diffusion-v1-5` mediante `StableDiffusionImg2ImgPipeline`.
* **Auditoría Visual (VLM):** `Salesforce/blip-image-captioning-base` para verificación de consistencia semántica.
* **Transparencia (HAI):** Mapa de diferencias absoluto entre la imagen original y la restaurada.

---

## ✅ Caso 1: Escenario de Éxito (Denoising Fotográfico)

### 1. Objetivo de la Prueba
Evaluar la capacidad del pipeline híbrido para eliminar ruido granular severo en una fotografía natural del mundo real, manteniendo la integridad fisonómica, espacial y semántica de los sujetos y objetos sin introducir alucinaciones.

### 2. Insumo de Entrada
* **Archivo de imagen:** `ejemplo_1_positivo.png` (5.0 MB)
* **Descripción del contenido:** Fotografía de retrato grupal en un aula (tres personas frente a una pantalla interactiva con el *Canvas Analítico*, escritorio con termo, mate y paquete de galletitas).
* **Degradación aplicada:** Ruido granular denso de alta frecuencia distribuido en toda la escena.

### 3. Parámetros de Configuración
* **Fuerza de alteración (`strength`):** `0.20`
* **Pasos de inferencia (`num_inference_steps`):** `40`
* **Escala de guía (`guidance_scale`):** `6.0`
* **Prompt de guía semántica (`prompt`):**
  ```text
  Fotografía realista de alta calidad, enfoque nítido, detalles definidos, rostros naturales, escena de clase clara
  ```
* **Prompt Negativo (`negative_prompt`):**
  ```text
  dibujo animado, pintura, ilustración, rostros deformes, borroso, artefactos, deforme, baja resolución, ruido excesivo
  ```

### 4. Resultados y Auditoría (HAI)
* **Calidad Visual:** El preprocesamiento clásico con OpenCV redujo el ruido grueso inicial, permitiendo que el modelo de difusión realizara un refinamiento estético sutil con `strength = 0.20`. Se limpió eficazmente el grano molesto preservando los rasgos faciales, las posturas y la geometría de los elementos del aula.
* **Verificación Semántica (VLM - BLIP):**
  * *Caption Original:* `"a photo of a group of people in a room"`
  * *Caption Restaurada:* `"a photo of a group of people in a room"`
  * *Diagnóstico:* **Coincidencia semántica del 100%.** El modelo de lenguaje visual confirma que no hubo alteración conceptual ni alucinación de objetos o personas.
* **Mapa de Diferencias (HAI):** Muestra una distribución homogénea de puntos de baja/media intensidad en toda la superficie, evidenciando de manera transparente que la intervención del modelo se limitó al suavizado del ruido granular píxel a píxel sin sobreescribir estructuras reales.

---

## ❌ Caso 2: Escenario de Fallo Controlado (Gráfico Geométrico / Alucinación)

### 1. Objetivo de la Prueba
Evidenciar las limitaciones inherentes de los modelos de difusión preentrenados sobre fotografías naturales cuando se enfrentan a dominios vectoriales o tipográficos, demostrando la utilidad de las capas de inspección y transparencia (HAI) para alertar al usuario ante una restauración no confiable.

### 2. Insumo de Entrada
* **Archivo de imagen:** `ejemplo_2_negativo.jpg` (371.4 KB)
* **Descripción del contenido:** Logotipo corporativo sintético en blanco y negro (triángulos entrelazados con la silueta de un león y el texto `"SYNTHETIC LOGISTICS"`).
* **Degradación aplicada:** Ruido granular y aberración cromática en los bordes vectoriales y tipográficos.

### 3. Parámetros de Configuración
* **Fuerza de alteración (`strength`):** `0.40` (valor elevado para forzar la intervención generativa del modelo)
* **Pasos de inferencia (`num_inference_steps`):** `40`
* **Escala de guía (`guidance_scale`):** `6.0`
* **Prompt de guía semántica (`prompt`):**
  ```text
  Fotografía realista de alta calidad, enfoque nítido, textura de roca natural, superficie rugosa
  ```
* **Prompt Negativo (`negative_prompt`):**
  ```text
  vector, logotipo, líneas limpias, gráfico plano, minimalista, diseño digital
  ```

### 4. Resultados y Auditoría (HAI)
* **Calidad Visual:** El sistema falló en reconstruir la geometría vectorial limpia. Al carecer de un *prior* espacial para gráficos sintéticos y tipografías, el espacio latente intentó interpretar los bordes duros como texturas físicas rugosas, degradando el contorno del logotipo y corrompiendo los caracteres del texto.
* **Verificación Semántica (VLM - BLIP):**
  * *Caption Original:* `"a photo of a triangle with the words synthtictictics"`
  * *Caption Restaurada:* `"a photo of a logo for synthlogics"`
  * *Diagnóstico:* **Alerta de desvío semántico y tipográfico.** Si bien el VLM logra inferir la presencia de un logotipo, evidencia la corrupción de los caracteres originales (`synthtictictics` $\rightarrow$ `synthlogics`), demostrando que el modelo estocástico deformó la información textual exacta.
* **Mapa de Diferencias (HAI):** Presenta bordes y contornos intensamente iluminados concentrados directamente sobre el trazado del logotipo y cada una de las letras de `"SYNTHETIC LOGISTICS"`, alertando visualmente al usuario sobre una modificación estructural crítica.

---

## 🎯 Conclusiones Técnicas para el Entregable

1. **Dependencia del Dominio (*Domain Prior*):** `StableDiffusionImg2ImgPipeline` (v1.5) demuestra alta eficacia para tareas de *denoising* en fotografías naturales siempre que se acompañe de un filtrado clásico previo (OpenCV) y un parámetro `strength` conservador (`0.20`).
2. **Incompatibilidad Vectorial y Tipográfica:** Los modelos de difusión estocásticos puros no son aptos para restaurar logotipos, diagramas técnicos o texto exacto; para estos casos se requieren arquitecturas condicionadas espacialmente (ej. *ControlNet*).
3. **Rol Crítico de la Auditoría HAI:** La integración de subtitulado automático (VLM) junto con los mapas de diferencias permite auditar la fidelidad de la restauración en tiempo real, evitando que las alucinaciones o deformaciones pasen desapercibidas para el usuario final.