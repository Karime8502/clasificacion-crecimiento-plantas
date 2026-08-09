# Clasificación de etapas de crecimiento de plantas

## Descripción
Sistema basado en redes neuronales convolucionales para clasificar
imágenes de plantas de frijol en tres etapas de desarrollo: temprana,
intermedia y avanzada. Las imágenes fueron capturadas con un montaje
propio (Raspberry Pi + cámara) instalado en la granja de la
universidad, complementado con fotografías desde teléfono móvil.

**El dataset es propio**: no se utilizaron datasets públicos como
fuente principal de entrenamiento.

## Tecnologías
- Python
- TensorFlow / Keras
- Roboflow (organización, etiquetado y preprocesamiento del dataset propio)
- NumPy, Matplotlib, Scikit-learn

## Estructura del repositorio
```
clasificacion-crecimiento-plantas/
├── codigo/
│   ├── preparar_datos.py     # carga y preprocesa el dataset
│   ├── modelo_cnn.py         # CNN desde cero
│   ├── modelo_mobilenet.py   # MobileNetV2 con Transfer Learning
│   ├── entrenar.py           # entrenamiento (ambos modelos)
│   ├── evaluar.py            # métricas y matriz de confusión
│   └── predecir.py           # inferencia sobre una imagen nueva
├── modelos/                  # pesos entrenados (.keras)
├── resultados/                # curvas, matriz de confusión, reportes
├── notebooks/
│   └── entrenamiento.ipynb   # notebook de Colab con el flujo completo
├── requirements.txt
└── DOCUMENTACION.md
```

## Dataset
El dataset fue construido por el equipo (Raspberry Pi + cámara,
complementado con teléfono móvil) y gestionado mediante Roboflow para
el etiquetado, preprocesamiento (resize 224x224, normalización) y
generación de los splits train/valid/test.

Clases:
- `temprana`
- `intermedia`
- `avanzada`

El dataset completo (imágenes) se mantiene en Roboflow / Google Drive
y no se incluye completo en este repositorio por su tamaño. Enlace:
_pendiente de agregar_.

## Arquitecturas
1. **CNN desde cero** (`modelo_cnn.py`): Conv2D/MaxPooling x3 + Dense + Dropout.
2. **MobileNetV2 con Transfer Learning** (`modelo_mobilenet.py`): base
   preentrenada en ImageNet (congelada) + capa densa de clasificación.

Ambos modelos usan Data Augmentation y Dropout, técnicas de
generalización obligatorias según el enunciado.

## Cómo ejecutar
Ver `notebooks/entrenamiento.ipynb` para el flujo completo en Google
Colab (recomendado, incluye GPU gratuita). Alternativamente, de forma
local:

```bash
pip install -r requirements.txt

python codigo/entrenar.py --modelo cnn --dataset_dir dataset --epocas 25
python codigo/entrenar.py --modelo mobilenet --dataset_dir dataset --epocas 25

python codigo/evaluar.py --modelo modelos/modelo_cnn.keras --dataset_dir dataset
python codigo/evaluar.py --modelo modelos/modelo_mobilenet.keras --dataset_dir dataset

python codigo/predecir.py --modelo modelos/modelo_mobilenet.keras --imagen ruta/a/foto.jpg
```

## Evaluación
Se utilizan las métricas exigidas por el profesor para tareas de
clasificación:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Matriz de confusión

Los resultados (gráficas y reportes) se guardan automáticamente en
`resultados/`.
