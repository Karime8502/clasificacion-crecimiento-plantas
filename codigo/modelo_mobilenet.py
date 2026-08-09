"""
modelo_mobilenet.py
--------------------
Segundo modelo del proyecto: Transfer Learning sobre MobileNetV2
preentrenada en ImageNet. La literatura revisada en el estado del
arte (p. ej. wheat growth stages con MobDenNet, transfer learning en
agricultura) sugiere que este enfoque suele superar a una CNN
entrenada desde cero cuando el dataset propio es pequeño.

Arquitectura:
    Imagen 224x224x3
      -> MobileNetV2 (congelada, pesos de ImageNet)
      -> GlobalAveragePooling2D
      -> Dropout(0.3)
      -> Dense(3) -> Softmax
"""

import tensorflow as tf
from tensorflow.keras import layers

from preparar_datos import crear_augmentation

NUM_CLASES = 3


def crear_mobilenet(usar_augmentation=True, fine_tune=False, fine_tune_desde=100):
    """
    Construye el modelo de Transfer Learning basado en MobileNetV2.

    Parámetros
    ----------
    usar_augmentation : bool
        Antepone Data Augmentation antes de la red base.
    fine_tune : bool
        Si es True, descongela las últimas capas de MobileNetV2 para
        un ajuste fino (fine-tuning) tras un primer entrenamiento con
        la base congelada. Recomendado solo después de que el modelo
        con la base congelada ya haya convergido.
    fine_tune_desde : int
        Índice de capa a partir de la cual se descongela la red base,
        si fine_tune=True. Las primeras capas (más cercanas a la
        entrada) capturan bordes y texturas genéricas y conviene
        dejarlas congeladas.
    """
    # MobileNetV2 se utiliza como extractor de características,
    # utilizando pesos previamente entrenados con ImageNet.
    # include_top=False descarta la capa de clasificación original
    # (1000 clases de ImageNet) para reemplazarla por la nuestra (3 clases).
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
        # Con la base congelada, solo se entrenan las capas nuevas
        # (GlobalAveragePooling + Dense). Esto es rápido y funciona
        # bien con pocos datos, que es el caso del dataset propio.
        base_model.trainable = False

    entradas = tf.keras.Input(shape=(224, 224, 3))
    x = entradas

    if usar_augmentation:
        x = crear_augmentation()(x)

    # MobileNetV2 espera entradas preprocesadas en el rango [-1, 1],
    # no [0, 1] como la CNN propia.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    salidas = layers.Dense(NUM_CLASES, activation="softmax")(x)

    modelo = tf.keras.Model(entradas, salidas, name="mobilenetv2_transfer_learning")
    return modelo


if __name__ == "__main__":
    modelo = crear_mobilenet()
    modelo.summary()
