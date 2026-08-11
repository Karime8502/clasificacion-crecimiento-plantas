"""
modelo_cnn.py  (versión DETECCIÓN DE OBJETOS)
------------------------------------------------
CNN construida desde cero, con DOS salidas (modelo multi-tarea):

  - clase_output: a qué etapa de madurez pertenece la planta (softmax, 3 clases)
  - caja_output:  dónde está la planta en la imagen (4 números: xmin,ymin,xmax,ymax
                   normalizados entre 0 y 1)

Ambas salidas comparten el mismo "tronco" convolucional (extracción
de características) y luego se separan en dos "cabezas" (heads)
independientes.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from preparar_datos import crear_augmentation

NUM_CLASES = 3


def crear_cnn(usar_augmentation=True):
    entradas = tf.keras.Input(shape=(224, 224, 3), name="imagen")
    x = entradas

    if usar_augmentation:
        x = crear_augmentation()(x)

    x = layers.Rescaling(1. / 255)(x)

    # --- Tronco convolucional compartido ---
    x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)  # regularización obligatoria

    # --- Cabeza de clasificación (madurez de la planta) ---
    # 0 = avanzada, 1 = intermedia, 2 = temprana (orden de CLASES en preparar_datos.py)
    clase_output = layers.Dense(NUM_CLASES, activation="softmax", name="clase_output")(x)

    # --- Cabeza de regresión de caja (localización de la planta) ---
    # sigmoid porque las coordenadas están normalizadas entre 0 y 1
    caja_output = layers.Dense(4, activation="sigmoid", name="caja_output")(x)

    modelo = models.Model(inputs=entradas, outputs=[clase_output, caja_output], name="cnn_deteccion")
    return modelo


if __name__ == "__main__":
    modelo = crear_cnn()
    modelo.summary()
