"""
modelo_mobilenet.py  (versión DETECCIÓN DE OBJETOS)
-------------------------------------------------------
Transfer Learning sobre MobileNetV2 (pesos de ImageNet), con las
mismas dos salidas del modelo CNN: clase_output (madurez) y
caja_output (ubicación de la planta).
"""

import tensorflow as tf
from tensorflow.keras import layers

from preparar_datos import crear_augmentation

NUM_CLASES = 3


def crear_mobilenet(usar_augmentation=True, fine_tune=False, fine_tune_desde=100):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )

    if fine_tune:
        base_model.trainable = True
        for capa in base_model.layers[:fine_tune_desde]:
            capa.trainable = False
    else:
        base_model.trainable = False

    entradas = tf.keras.Input(shape=(224, 224, 3), name="imagen")
    x = entradas

    if usar_augmentation:
        x = crear_augmentation()(x)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)  # regularización obligatoria

    # --- Cabeza de clasificación ---
    clase_output = layers.Dense(NUM_CLASES, activation="softmax", name="clase_output")(x)

    # --- Cabeza de regresión de caja ---
    caja_output = layers.Dense(4, activation="sigmoid", name="caja_output")(x)

    modelo = tf.keras.Model(entradas, [clase_output, caja_output], name="mobilenetv2_deteccion")
    return modelo


if __name__ == "__main__":
    modelo = crear_mobilenet()
    modelo.summary()
