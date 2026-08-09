"""
preparar_datos.py
------------------
Carga las imágenes del dataset (exportado desde Roboflow en formato de
carpetas por clase: train/valid/test) y las convierte en objetos
tf.data.Dataset listos para entrenar.

Estructura esperada en disco:

dataset/
├── train/
│   ├── temprana/
│   ├── intermedia/
│   └── avanzada/
├── valid/
│   ├── temprana/
│   ├── intermedia/
│   └── avanzada/
└── test/
    ├── temprana/
    ├── intermedia/
    └── avanzada/
"""

import tensorflow as tf

# Tamaño utilizado para normalizar todas las imágenes
# antes de introducirlas en la red neuronal. MobileNetV2 y la CNN
# propia usan el mismo tamaño de entrada para poder compararse.
IMG_SIZE = (224, 224)

# Tamaño de lote (batch). Con datasets pequeños (pocas decenas/cientos
# de imágenes) conviene un batch pequeño para tener más pasos de
# actualización por época.
BATCH_SIZE = 16

# Nombres de las clases en el orden en que Keras las asigna
# alfabéticamente al leer las carpetas: avanzada=0, intermedia=1, temprana=2.
# (Se confirma e imprime en tiempo de ejecución con class_names más abajo.)
CLASSES = ["avanzada", "intermedia", "temprana"]


def cargar_datasets(dataset_dir="dataset"):
    """
    Carga los tres splits (train, valid, test) desde disco.

    Parámetros
    ----------
    dataset_dir : str
        Ruta a la carpeta que contiene train/, valid/ y test/.

    Retorna
    -------
    train_dataset, validation_dataset, test_dataset : tf.data.Dataset
    """
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        f"{dataset_dir}/train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        f"{dataset_dir}/valid",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        f"{dataset_dir}/test",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("Clases detectadas:", train_dataset.class_names)

    # prefetch mejora el rendimiento: mientras la GPU entrena un lote,
    # la CPU ya está preparando el siguiente.
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset


def crear_augmentation():
    """
    Define las capas de Data Augmentation exigidas por el profesor
    como técnica obligatoria de generalización.

    Se aplican solo durante el entrenamiento (Keras las desactiva
    automáticamente en modo de evaluación/inferencia).
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.15),
        tf.keras.layers.RandomContrast(0.15),
    ], name="data_augmentation")


if __name__ == "__main__":
    train_ds, val_ds, test_ds = cargar_datasets()
    print("Lotes de entrenamiento:", tf.data.experimental.cardinality(train_ds).numpy())
