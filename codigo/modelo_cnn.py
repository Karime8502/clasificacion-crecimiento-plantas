"""
modelo_cnn.py
-------------
Arquitectura CNN construida y entrenada desde cero (sin pesos
preentrenados). Sirve como línea base para comparar contra el modelo
con Transfer Learning (modelo_mobilenet.py).

Arquitectura:
    Imagen 224x224x3
      -> Conv2D(32) -> MaxPooling
      -> Conv2D(64) -> MaxPooling
      -> Conv2D(128) -> MaxPooling
      -> Flatten
      -> Dense(128) -> Dropout(0.3)
      -> Dense(3) -> Softmax
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from preparar_datos import crear_augmentation

NUM_CLASES = 3


def crear_cnn(usar_augmentation=True):
    """
    Construye la CNN desde cero.

    Parámetros
    ----------
    usar_augmentation : bool
        Si es True, antepone las capas de Data Augmentation
        (obligatorias según el enunciado del profesor).
    """
    capas = []

    if usar_augmentation:
        capas.append(crear_augmentation())

    # Rescaling normaliza los píxeles de [0, 255] a [0, 1],
    # lo cual ayuda a que el entrenamiento converja más rápido y
    # de forma más estable.
    capas.append(layers.Rescaling(1. / 255))

    capas += [
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),

        # Dropout: apaga aleatoriamente el 30% de las neuronas en cada
        # paso de entrenamiento. Es la segunda técnica de
        # regularización obligatoria: reduce el sobreajuste, algo
        # especialmente importante porque el dataset propio es pequeño.
        layers.Dropout(0.3),

        # La capa de salida contiene tres neuronas,
        # una para cada etapa de crecimiento:
        # 0 = avanzada, 1 = intermedia, 2 = temprana
        # (orden alfabético que usa image_dataset_from_directory)
        layers.Dense(NUM_CLASES, activation="softmax"),
    ]

    modelo = models.Sequential(capas, name="cnn_desde_cero")
    # build() fuerza a Keras a construir las formas de las capas
    # para poder imprimir el resumen antes de entrenar.
    modelo.build(input_shape=(None, 224, 224, 3))
    return modelo


if __name__ == "__main__":
    modelo = crear_cnn()
    modelo.summary()
