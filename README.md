# Proyecto de Análisis de Sentimientos IMDB

Este proyecto implementa un sistema de análisis de sentimientos para reseñas de películas utilizando Redes Neuronales Recurrentes (RNNs), específicamente arquitecturas LSTM y GRU. Además, proporciona una interfaz web interactiva desarrollada con FastAPI.

## Configuración

1.  **Instalar Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Datos:**
    Asegúrate de que la carpeta `aclImdb` esté en el directorio raíz. Debe contener las carpetas `train` y `test`, así como el archivo `imdb.vocab`.

## Uso
Para ejecutar la interfaz web, utiliza el siguiente comando:
```bash
python app.py
```

## Re-evaluación de Modelos
Para re-entrenar los modelos y evaluarlos nuevamente, sigue estos pasos:

### 1. Entrenar los Modelos
Entrena ambos modelos (LSTM y GRU), compara su rendimiento y guarda el mejor.
```bash
python train.py
```
*   Generará el archivo `best_model.keras` y gráficas del historial de entrenamiento.
*   Guardará los resultados comparativos en `model_comparison.csv`.

### 2. Evaluar el Mejor Modelo
Ejecuta una evaluación detallada en el conjunto de prueba (Exactitud, Precisión, Recall, F1, Matriz de Confusión).
```bash
python evaluate.py
```
*   Imprimirá las métricas en consola y guardará la matriz de confusión como `confusion_matrix.png`.
*   Mostrará ejemplos de predicciones correctas e incorrectas.

### 3. Ejecutar la Interfaz Web
Inicia el servidor FastAPI para usar el modelo de forma interactiva.
```bash
python app.py
```
*   Abre tu navegador y ve a `http://127.0.0.1:8000`.
*   Ingresa una reseña de película (en inglés) para ver la predicción del sentimiento.

## Estructura del Proyecto

*   `data_loader.py`: Utilidades para cargar los datos de texto y crear la capa de vectorización.
*   `models.py`: Definición de las arquitecturas de redes neuronales (LSTM y GRU).
*   `train.py`: Script principal para entrenar los modelos y guardar el mejor.
*   `evaluate.py`: Script para evaluar el modelo guardado y analizar errores.
*   `app.py`: Aplicación web con FastAPI.
*   `requirements.txt`: Lista de dependencias de Python.
