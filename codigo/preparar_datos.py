"""
preparar_datos.py  (versión DETECCIÓN DE OBJETOS — formato YOLO)
--------------------------------------------------------------------
Carga imágenes + su caja delimitadora (bounding box) + su clase de
madurez, a partir de un dataset exportado de Roboflow en formato
YOLO (un .txt por imagen).

Estructura esperada (la que ya tienen en Drive):

dataset/
├── data.yaml
├── train/
│   ├── images/
│   │   ├── foto1.jpg
│   │   └── ...
│   └── labels/
│       ├── foto1.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

Cada .txt tiene una línea por objeto (aquí siempre una sola, porque
cada foto tiene una sola planta):
    clase_id  x_centro  y_centro  ancho  alto
todo normalizado entre 0 y 1 respecto al tamaño de la imagen.

El orden de las clases (qué número es cada nombre) se lee del
data.yaml, en la llave "names".
"""

import os

import tensorflow as tf
import yaml

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Valor de respaldo; se sobreescribe de verdad al llamar cargar_datasets(),
# que lee el orden real desde dataset/data.yaml.
CLASES = ["avanzada", "intermedia", "temprana"]


def _leer_nombres_clases(dataset_dir):
    ruta_yaml = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(ruta_yaml):
        raise FileNotFoundError(f"No se encontró {ruta_yaml}.")
    with open(ruta_yaml, "r") as f:
        data = yaml.safe_load(f)
    return data["names"]


def _leer_split(dataset_dir, split):
    carpeta_imagenes = os.path.join(dataset_dir, split, "images")
    carpeta_labels = os.path.join(dataset_dir, split, "labels")

    if not os.path.isdir(carpeta_imagenes):
        raise FileNotFoundError(f"No se encontró la carpeta {carpeta_imagenes}")

    rutas, etiquetas, cajas = [], [], []

    for nombre_archivo in sorted(os.listdir(carpeta_imagenes)):
        if not nombre_archivo.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        nombre_base, _ = os.path.splitext(nombre_archivo)
        ruta_label = os.path.join(carpeta_labels, nombre_base + ".txt")

        if not os.path.exists(ruta_label):
            print(f"Aviso: sin etiqueta para {nombre_archivo}, se omite.")
            continue

        with open(ruta_label, "r") as f:
            lineas = [l.strip() for l in f if l.strip()]

        if not lineas:
            print(f"Aviso: {ruta_label} está vacío, se omite.")
            continue

        # Cada foto tiene una sola planta -> se toma solo la primera línea.
        valores = lineas[0].split()
        clase_id = int(valores[0])
        coords = list(map(float, valores[1:]))

        if len(coords) == 4:
            # Formato bounding box simple: x_centro, y_centro, ancho, alto
            x_c, y_c, ancho, alto = coords
            xmin = x_c - ancho / 2
            ymin = y_c - alto / 2
            xmax = x_c + ancho / 2
            ymax = y_c + alto / 2
        else:
            # Formato polígono (segmentación): pares x1,y1, x2,y2, ...
            # (así quedó exportado porque en Roboflow se etiquetó con
            # la herramienta de contorno/polígono, no de caja).
            # Se calcula la caja rectangular que envuelve el polígono.
            xs = coords[0::2]
            ys = coords[1::2]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

        rutas.append(os.path.join(carpeta_imagenes, nombre_archivo))
        etiquetas.append(clase_id)
        cajas.append([xmin, ymin, xmax, ymax])

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
    global CLASES
    CLASES = _leer_nombres_clases(dataset_dir)

    train_rutas, train_y, train_boxes = _leer_split(dataset_dir, "train")
    val_rutas, val_y, val_boxes = _leer_split(dataset_dir, "valid")
    test_rutas, test_y, test_boxes = _leer_split(dataset_dir, "test")

    print(f"Imágenes -> train: {len(train_rutas)}  valid: {len(val_rutas)}  test: {len(test_rutas)}")
    print("Clases (orden del data.yaml):", CLASES)

    train_dataset = _construir_dataset(train_rutas, train_y, train_boxes, entrenamiento=True)
    validation_dataset = _construir_dataset(val_rutas, val_y, val_boxes)
    test_dataset = _construir_dataset(test_rutas, test_y, test_boxes)

    return train_dataset, validation_dataset, test_dataset


def crear_augmentation():
    """
    Data Augmentation SOLO sobre color (brillo, contraste). No se
    aplican transformaciones geométricas porque moverían la caja
    delimitadora fuera de su posición real.
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
