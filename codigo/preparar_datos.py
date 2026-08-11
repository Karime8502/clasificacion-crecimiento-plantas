"""
preparar_datos.py  (versión DETECCIÓN DE OBJETOS)
----------------------------------------------------
Carga imágenes + su caja delimitadora (bounding box) + su clase de
madurez, a partir de un dataset exportado de Roboflow en formato
"CSV" (una fila por imagen, porque cada foto tiene UNA sola planta).

Estructura esperada (la que genera Roboflow al exportar en CSV):

dataset/
├── train/
│   ├── _annotations.csv
│   ├── foto1.jpg
│   ├── foto2.jpg
│   └── ...
├── valid/
│   ├── _annotations.csv
│   └── ...
└── test/
    ├── _annotations.csv
    └── ...

Columnas esperadas en _annotations.csv:
    filename, width, height, class, xmin, ymin, xmax, ymax
"""

import os

import pandas as pd
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Orden fijo de clases (debe coincidir con cómo se compilan las
# métricas en entrenar.py y evaluar.py). Ajustar si en Roboflow
# quedaron con otro nombre exacto.
CLASES = ["avanzada", "intermedia", "temprana"]
CLASE_A_INDICE = {nombre: i for i, nombre in enumerate(CLASES)}


def _leer_split(dataset_dir, split):
    """
    Lee el CSV de un split (train/valid/test) y arma listas paralelas
    de: rutas de imagen, etiqueta de clase (entero) y caja normalizada
    [xmin, ymin, xmax, ymax] en rango [0, 1] (relativa al ancho/alto
    original de cada imagen).
    """
    carpeta = os.path.join(dataset_dir, split)
    ruta_csv = os.path.join(carpeta, "_annotations.csv")

    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(
            f"No se encontró {ruta_csv}. Verifiquen que exportaron el "
            f"dataset desde Roboflow en formato 'CSV' (no YOLO/COCO)."
        )

    df = pd.read_csv(ruta_csv)

    # Algunas exportaciones de Roboflow usan mayúsculas o nombres
    # ligeramente distintos; normalizamos por si acaso.
    df.columns = [c.strip().lower() for c in df.columns]

    rutas, etiquetas, cajas = [], [], []

    for _, fila in df.iterrows():
        nombre_clase = str(fila["class"]).strip().lower()
        if nombre_clase not in CLASE_A_INDICE:
            print(f"Aviso: clase desconocida '{nombre_clase}' en {fila['filename']}, se omite.")
            continue

        ruta_imagen = os.path.join(carpeta, fila["filename"])
        if not os.path.exists(ruta_imagen):
            print(f"Aviso: no se encontró la imagen {ruta_imagen}, se omite.")
            continue

        ancho = float(fila["width"])
        alto = float(fila["height"])

        # Normalizamos la caja a [0, 1] para que no dependa de la
        # resolución original de cada foto (pueden venir de cámara y
        # celular con tamaños distintos).
        caja_normalizada = [
            float(fila["xmin"]) / ancho,
            float(fila["ymin"]) / alto,
            float(fila["xmax"]) / ancho,
            float(fila["ymax"]) / alto,
        ]

        rutas.append(ruta_imagen)
        etiquetas.append(CLASE_A_INDICE[nombre_clase])
        cajas.append(caja_normalizada)

    return rutas, etiquetas, cajas


def _cargar_imagen(ruta, etiqueta, caja):
    imagen = tf.io.read_file(ruta)
    imagen = tf.image.decode_jpeg(imagen, channels=3)
    imagen = tf.image.resize(imagen, IMG_SIZE)
    return imagen, {"clase_output": etiqueta, "caja_output": caja}


def _construir_dataset(rutas, etiquetas, cajas, entrenamiento=False):
    ds = tf.data.Dataset.from_tensor_slices((rutas, etiquetas, cajas))
    ds = ds.map(_cargar_imagen, num_parallel_calls=tf.data.AUTOTUNE)

    if entrenamiento:
        ds = ds.shuffle(buffer_size=max(len(rutas), 1), seed=42)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def cargar_datasets(dataset_dir="dataset"):
    """
    Retorna train_dataset, validation_dataset, test_dataset listos
    para .fit()/.evaluate(), cada elemento con forma:
        (imagen, {"clase_output": etiqueta, "caja_output": [xmin,ymin,xmax,ymax]})
    """
    train_rutas, train_y, train_boxes = _leer_split(dataset_dir, "train")
    val_rutas, val_y, val_boxes = _leer_split(dataset_dir, "valid")
    test_rutas, test_y, test_boxes = _leer_split(dataset_dir, "test")

    print(f"Imágenes -> train: {len(train_rutas)}  valid: {len(val_rutas)}  test: {len(test_rutas)}")
    print("Clases:", CLASES)

    train_dataset = _construir_dataset(train_rutas, train_y, train_boxes, entrenamiento=True)
    validation_dataset = _construir_dataset(val_rutas, val_y, val_boxes)
    test_dataset = _construir_dataset(test_rutas, test_y, test_boxes)

    return train_dataset, validation_dataset, test_dataset


def crear_augmentation():
    """
    Data Augmentation SOLO sobre color (brillo, contraste). No se
    aplican transformaciones geométricas (rotación, volteo, zoom)
    porque estas moverían la planta dentro de la imagen y dejarían
    la caja delimitadora (xmin,ymin,xmax,ymax) desactualizada frente
    a la posición real de la planta, salvo que también se recalculen
    las coordenadas de la caja en cada transformación (no implementado
    aquí por simplicidad). Si más adelante quieren agregar rotación,
    avísenme y lo hacemos con las coordenadas de la caja incluidas.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomBrightness(0.15),
        tf.keras.layers.RandomContrast(0.15),
    ], name="data_augmentation")


if __name__ == "__main__":
    train_ds, val_ds, test_ds = cargar_datasets()
    for imagenes, etiquetas in train_ds.take(1):
        print("Forma de imagen:", imagenes.shape)
        print("Forma clase_output:", etiquetas["clase_output"].shape)
        print("Forma caja_output:", etiquetas["caja_output"].shape)
