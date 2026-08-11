"""
predecir.py  (versión DETECCIÓN DE OBJETOS)
------------------------------------------------
Carga un modelo entrenado y, para UNA imagen nueva:
  1. dice en qué etapa de madurez está la planta,
  2. dibuja la caja delimitadora predicha alrededor de la planta y
     guarda la imagen resultante.

Uso:
    python predecir.py --modelo modelos/modelo_mobilenet.keras --imagen planta.jpg
"""

import argparse
import os

import numpy as np
import tensorflow as tf
import yaml
from PIL import Image, ImageDraw

IMG_SIZE = (224, 224)
CLASES_POR_DEFECTO = ["avanzada", "intermedia", "temprana"]


def _obtener_clases(dataset_dir=None):
    if dataset_dir is None:
        return CLASES_POR_DEFECTO
    ruta_yaml = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(ruta_yaml):
        print(f"Aviso: no se encontró {ruta_yaml}, usando orden por defecto.")
        return CLASES_POR_DEFECTO
    with open(ruta_yaml, "r") as f:
        data = yaml.safe_load(f)
    return data["names"]


def predecir_imagen(ruta_modelo, ruta_imagen, dataset_dir=None, carpeta_salida="resultados"):
    clases = _obtener_clases(dataset_dir)
    modelo = tf.keras.models.load_model(ruta_modelo)

    imagen_original = Image.open(ruta_imagen).convert("RGB")
    imagen_redimensionada = imagen_original.resize(IMG_SIZE)
    arreglo = tf.keras.utils.img_to_array(imagen_redimensionada)
    lote = np.expand_dims(arreglo, axis=0)

    clase_probs, caja_pred = modelo.predict(lote, verbose=0)
    clase_probs = clase_probs[0]
    caja_pred = caja_pred[0]  # [xmin, ymin, xmax, ymax] normalizados 0-1

    indice_predicho = int(np.argmax(clase_probs))
    etapa = clases[indice_predicho]

    print("Resultado:")
    print(f"Etapa: {etapa.upper()}")
    print("Probabilidades:")
    for clase, prob in zip(clases, clase_probs):
        print(f"  {clase.capitalize()}: {prob:.2f}")
    print(f"Caja (normalizada): {caja_pred}")

    ancho_orig, alto_orig = imagen_original.size
    xmin = caja_pred[0] * ancho_orig
    ymin = caja_pred[1] * alto_orig
    xmax = caja_pred[2] * ancho_orig
    ymax = caja_pred[3] * alto_orig

    imagen_dibujada = imagen_original.copy()
    draw = ImageDraw.Draw(imagen_dibujada)
    draw.rectangle([xmin, ymin, xmax, ymax], outline="lime", width=4)
    draw.text((xmin, max(0, ymin - 20)), f"{etapa} ({clase_probs[indice_predicho]:.0%})", fill="lime")

    os.makedirs(carpeta_salida, exist_ok=True)
    nombre_salida = os.path.join(carpeta_salida, "prediccion_" + os.path.basename(ruta_imagen))
    imagen_dibujada.save(nombre_salida)
    print(f"\nImagen con la caja dibujada guardada en: {nombre_salida}")

    return etapa, clase_probs, caja_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predice etapa y ubicación de la planta en una imagen.")
    parser.add_argument("--modelo", required=True, help="Ruta al archivo .keras")
    parser.add_argument("--imagen", required=True, help="Ruta a la imagen a analizar")
    parser.add_argument("--dataset_dir", default=None,
                         help="Carpeta del dataset (para leer el orden real de clases desde data.yaml)")
    args = parser.parse_args()

    predecir_imagen(args.modelo, args.imagen, dataset_dir=args.dataset_dir)
