"""
generar_imagenes_aumentadas.py
--------------------------------
Genera copias NUEVAS de las imágenes del dataset, con variaciones
aleatorias de brillo, contraste, nitidez, rotación y zoom, y las
guarda como archivos .jpg nuevos en disco.

A diferencia del Data Augmentation "en memoria" que ya usa
entrenar.py (que varía las imágenes solo durante el entrenamiento,
sin guardar nada), este script SÍ crea archivos físicos nuevos.
Úsalo si quieren tener más imágenes reales en el dataset antes de
subirlo a Roboflow, o si alguna clase tiene muy pocas fotos frente a
las demás.

Estructura esperada de entrada (una carpeta por clase):

    dataset_original/
    ├── temprana/
    ├── intermedia/
    └── avanzada/

Salida (mismas carpetas, con originales + nuevas):

    dataset_aumentado/
    ├── temprana/       (originales + variantes nuevas)
    ├── intermedia/
    └── avanzada/

Uso:
    pip install pillow tqdm

    python generar_imagenes_aumentadas.py \
        --entrada dataset_original \
        --salida dataset_aumentado \
        --variantes_por_imagen 3
"""

import argparse
import os
import random

from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png")


def aplicar_transformaciones_aleatorias(imagen: Image.Image) -> Image.Image:
    """
    Aplica una combinación aleatoria de transformaciones a una imagen
    y devuelve una nueva imagen (la original no se modifica).
    """
    img = imagen.copy()

    # --- Rotación aleatoria pequeña (evita rotar tanto que la planta
    # quede "de cabeza", lo cual no pasaría en capturas reales) ---
    angulo = random.uniform(-15, 15)
    img = img.rotate(angulo, expand=False, fillcolor=(255, 255, 255))

    # --- Volteo horizontal aleatorio (una planta vista al espejo
    # sigue siendo una planta válida en esa etapa) ---
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # --- Brillo: entre 80% y 130% del original ---
    factor_brillo = random.uniform(0.8, 1.3)
    img = ImageEnhance.Brightness(img).enhance(factor_brillo)

    # --- Contraste: entre 85% y 125% ---
    factor_contraste = random.uniform(0.85, 1.25)
    img = ImageEnhance.Contrast(img).enhance(factor_contraste)

    # --- Nitidez: entre 70% (más borrosa, simula desenfoque de cámara)
    # y 180% (más nítida/enfocada) ---
    factor_nitidez = random.uniform(0.7, 1.8)
    img = ImageEnhance.Sharpness(img).enhance(factor_nitidez)

    # --- Saturación de color: entre 85% y 120% ---
    factor_color = random.uniform(0.85, 1.2)
    img = ImageEnhance.Color(img).enhance(factor_color)

    # --- Zoom aleatorio (recorta un poco el centro y reescala) ---
    if random.random() < 0.5:
        ancho, alto = img.size
        factor_zoom = random.uniform(0.85, 0.98)
        nuevo_ancho, nuevo_alto = int(ancho * factor_zoom), int(alto * factor_zoom)
        izquierda = (ancho - nuevo_ancho) // 2
        arriba = (alto - nuevo_alto) // 2
        img = img.crop((izquierda, arriba, izquierda + nuevo_ancho, arriba + nuevo_alto))
        img = img.resize((ancho, alto))

    # --- Un desenfoque muy leve, ocasional, simulando cámara movida ---
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

    return img


def procesar_carpeta(entrada_dir: str, salida_dir: str, variantes_por_imagen: int):
    os.makedirs(salida_dir, exist_ok=True)

    clases = [c for c in os.listdir(entrada_dir) if os.path.isdir(os.path.join(entrada_dir, c))]
    print(f"Clases encontradas: {clases}")

    resumen = {}

    for clase in clases:
        carpeta_entrada = os.path.join(entrada_dir, clase)
        carpeta_salida = os.path.join(salida_dir, clase)
        os.makedirs(carpeta_salida, exist_ok=True)

        archivos = [
            f for f in os.listdir(carpeta_entrada)
            if f.lower().endswith(EXTENSIONES_VALIDAS)
        ]

        contador_nuevas = 0
        for archivo in tqdm(archivos, desc=f"Procesando '{clase}'"):
            ruta_original = os.path.join(carpeta_entrada, archivo)
            nombre_base, _ = os.path.splitext(archivo)

            try:
                imagen = Image.open(ruta_original).convert("RGB")
            except Exception as e:
                print(f"  Aviso: no se pudo abrir {archivo} ({e}), se omite.")
                continue

            # Copiar la original tal cual a la carpeta de salida
            imagen.save(os.path.join(carpeta_salida, archivo), quality=95)

            # Generar N variantes nuevas de esa imagen
            for i in range(variantes_por_imagen):
                variante = aplicar_transformaciones_aleatorias(imagen)
                nombre_variante = f"{nombre_base}_aug{i+1}.jpg"
                variante.save(os.path.join(carpeta_salida, nombre_variante), quality=95)
                contador_nuevas += 1

        resumen[clase] = {
            "originales": len(archivos),
            "nuevas": contador_nuevas,
            "total": len(archivos) + contador_nuevas,
        }

    print("\n=== Resumen ===")
    for clase, datos in resumen.items():
        print(f"{clase:12s} -> originales: {datos['originales']:4d}  "
              f"nuevas: {datos['nuevas']:4d}  total: {datos['total']:4d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera imágenes aumentadas y las guarda en disco.")
    parser.add_argument("--entrada", required=True, help="Carpeta con subcarpetas por clase (dataset original).")
    parser.add_argument("--salida", required=True, help="Carpeta donde se guardará el dataset aumentado.")
    parser.add_argument("--variantes_por_imagen", type=int, default=3,
                         help="Cuántas imágenes nuevas generar por cada imagen original.")
    args = parser.parse_args()

    random.seed(42)  # reproducibilidad: mismos resultados si se corre de nuevo
    procesar_carpeta(args.entrada, args.salida, args.variantes_por_imagen)
