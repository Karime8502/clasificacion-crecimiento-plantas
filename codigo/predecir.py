"""
predecir.py
-----------
Carga un modelo ya entrenado y predice la etapa de crecimiento de UNA
imagen nueva. Este script es el que demuestra que el modelo puede
usarse en la práctica (por ejemplo, en el video de sustentación).

Uso:
    python predecir.py --modelo modelos/modelo_mobilenet.keras --imagen planta.jpg
"""

import argparse

import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)
CLASES = ["avanzada", "intermedia", "temprana"]  # orden alfabético (igual al dataset)


def predecir_imagen(ruta_modelo, ruta_imagen):
    modelo = tf.keras.models.load_model(ruta_modelo)

    imagen = tf.keras.utils.load_img(ruta_imagen, target_size=IMG_SIZE)
    arreglo = tf.keras.utils.img_to_array(imagen)
    lote = np.expand_dims(arreglo, axis=0)  # el modelo espera un "lote" de imágenes

    probabilidades = modelo.predict(lote, verbose=0)[0]
    indice_predicho = int(np.argmax(probabilidades))

    print("Resultado:")
    print(f"Etapa: {CLASES[indice_predicho].upper()}")
    print("Probabilidades:")
    for clase, prob in zip(CLASES, probabilidades):
        print(f"  {clase.capitalize()}: {prob:.2f}")

    return CLASES[indice_predicho], probabilidades


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predice la etapa de crecimiento de una imagen.")
    parser.add_argument("--modelo", required=True, help="Ruta al archivo .keras")
    parser.add_argument("--imagen", required=True, help="Ruta a la imagen a clasificar")
    args = parser.parse_args()

    predecir_imagen(args.modelo, args.imagen)
