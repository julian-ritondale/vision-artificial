## TP4 - Detección de Objetos con YOLOv8

Detección de objetos utilizando un modelo YOLOv8n personalizado entrenado para detectar 4 clases:
- `Person` (Persona)
- `Car` (Auto)
- `Airplane` (Avión)
- `Tank` (Tanque)

Las etiquetas y configuración del dataset se encuentran en `data.yaml` y los pesos del modelo en `best.pt`.

### Requisitos

Instalar las dependencias requeridas:

```bash
pip install -r tp4/requirements.txt
```

### Ejecución

Al ejecutar la predicción, se abre automáticamente una ventana gráfica interactiva con la imagen y las cajas asignadas (siguiendo el esquema de visualización con OpenCV de los otros TPs del repositorio).

Controles de la ventana:
- **Cualquier tecla**: pasa a la siguiente imagen o cierra la ventana.
- **`q` o `ESC`**: sale inmediatamente del visor.
- **Botón `(X)` de la ventana**: cierra la ventana.

#### Ejemplos de uso:

1. **Pasando una imagen individual (abre la ventana con detecciones):**
   ```bash
   python tp4/main.py tp4/vt4_main.jpg
   ```

2. **Pasando múltiples imágenes:**
   ```bash
   python tp4/main.py imagen1.jpg imagen2.png
   ```

3. **Pasando un directorio completo de imágenes:**
   ```bash
   python tp4/main.py --source /ruta/a/carpeta/
   ```

4. **Ajustando el umbral de confianza (`--conf`):**
   ```bash
   python tp4/main.py tp4/vt4_main.jpg --conf 0.5
   ```

5. **Modo sin ventana gráfica (headless / por lotes):**
   ```bash
   python tp4/main.py tp4/vt4_main.jpg --no-show
   ```

6. **Sin guardar los resultados en disco (`--no-save`):**
   ```bash
   python tp4/main.py tp4/vt4_main.jpg --no-save
   ```

7. **Modo interactivo (sin argumentos):**
   ```bash
   python tp4/main.py
   ```
   El script solicitará por teclado la ruta de la imagen o carpeta a procesar.

Por defecto, las imágenes con las anotaciones y bounding boxes también se guardan en `tp4/output/`.
