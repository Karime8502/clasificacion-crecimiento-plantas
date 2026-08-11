"""
evaluar.py  (versión DETECCIÓN DE OBJETOS)
----------------------------------------------
Evalúa un modelo multi-salida (clase_output + caja_output) sobre el
conjunto de prueba.

Para la CLASE calcula: Accuracy, Precision, Recall, F1-Score, matriz
de confusión (las métricas que pide el profesor para clasificación).

Para la CAJA calcula: IoU promedio (Intersection over Union), que es
la métrica estándar para saber qué tan bien ubicada quedó la caja
predicha frente a la real (las métricas de "Detección de Objetos"
que pide el profesor: IoU, además de Precision/Recall que ya se
calculan sobre la clase).

Uso:
    python evaluar.py --modelo modelos/modelo_mobilenet.keras --dataset_dir dataset
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import preparar_datos
from preparar_datos import cargar_datasets


def calcular_iou(caja_real, caja_predicha):
    """
    IoU (Intersection over Union) entre dos cajas [xmin, ymin, xmax, ymax]
    normalizadas. 1.0 = coinciden perfectamente, 0.0 = no se tocan.
    """
    xmin_inter = np.maximum(caja_real[:, 0], caja_predicha[:, 0])
    ymin_inter = np.maximum(caja_real[:, 1], caja_predicha[:, 1])
    xmax_inter = np.minimum(caja_real[:, 2], caja_predicha[:, 2])
    ymax_inter = np.minimum(caja_real[:, 3], caja_predicha[:, 3])

    ancho_inter = np.maximum(0, xmax_inter - xmin_inter)
    alto_inter = np.maximum(0, ymax_inter - ymin_inter)
    area_inter = ancho_inter * alto_inter

    area_real = (caja_real[:, 2] - caja_real[:, 0]) * (caja_real[:, 3] - caja_real[:, 1])
    area_pred = (caja_predicha[:, 2] - caja_predicha[:, 0]) * (caja_predicha[:, 3] - caja_predicha[:, 1])

    area_union = area_real + area_pred - area_inter
    iou = np.where(area_union > 0, area_inter / area_union, 0.0)
    return iou


def evaluar_modelo(ruta_modelo, dataset_dir="dataset", carpeta_salida="resultados"):
    os.makedirs(carpeta_salida, exist_ok=True)

    _, _, test_ds = cargar_datasets(dataset_dir)

    #  CÓDIGO CORREGIDO PARA REEMPLAZAR:
import os

# Si el usuario pasó solo 'cnn' o 'mobilenet', construimos la ruta real al archivo .keras
if ruta_modelo in ["cnn", "mobilenet"]:
    nombre_corto = ruta_modelo
    ruta_modelo = f"modelos/modelo_{nombre_corto}.keras"
    
    # Si estás ejecutando dentro de la subcarpeta /codigo, buscamos un nivel atrás
    if not os.path.exists(ruta_modelo):
        ruta_modelo = f"../modelos/modelo_{nombre_corto}.keras"

print(f"Cargando de forma segura el archivo del modelo desde: {ruta_modelo}")
modelo = tf.keras.models.load_model(ruta_modelo)

    y_true, y_pred = [], []
    cajas_true, cajas_pred = [], []

    for imagenes, etiquetas in test_ds:
        clase_pred, caja_pred = modelo.predict(imagenes, verbose=0)

        y_true.append(etiquetas["clase_output"].numpy())
        y_pred.append(np.argmax(clase_pred, axis=1))

        cajas_true.append(etiquetas["caja_output"].numpy())
        cajas_pred.append(caja_pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    cajas_true = np.concatenate(cajas_true)
    cajas_pred = np.concatenate(cajas_pred)

    # --- Métricas de clasificación (madurez) ---
    print("\n=== Reporte de clasificación (Accuracy, Precision, Recall, F1) ===")
    reporte = classification_report(y_true, y_pred, target_names=preparar_datos.CLASES)
    print(reporte)

    matriz = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=preparar_datos.CLASES)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Matriz de confusión - clase de madurez")
    plt.savefig(f"{carpeta_salida}/matriz_confusion.png", bbox_inches="tight")
    plt.close()

    # --- Métricas de detección (ubicación de la planta) ---
    iou_por_imagen = calcular_iou(cajas_true, cajas_pred)
    iou_promedio = float(np.mean(iou_por_imagen))
    # Porcentaje de cajas "aceptables": IoU >= 0.5 es el umbral estándar
    # usado en la literatura de detección de objetos.
    porcentaje_iou_50 = float(np.mean(iou_por_imagen >= 0.5)) * 100

    print(f"\n=== Métricas de localización (caja) ===")
    print(f"IoU promedio: {iou_promedio:.4f}")
    print(f"% de cajas con IoU >= 0.5: {porcentaje_iou_50:.1f}%")

    with open(f"{carpeta_salida}/reporte_clasificacion.txt", "w") as f:
        f.write(reporte)
        f.write(f"\nIoU promedio: {iou_promedio:.4f}\n")
        f.write(f"% de cajas con IoU >= 0.5: {porcentaje_iou_50:.1f}%\n")

    print(f"\nResultados guardados en {carpeta_salida}/")
    return reporte, matriz, iou_promedio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa un modelo de detección entrenado.")
    parser.add_argument("--modelo", required=True, help="Ruta al archivo .keras")
    parser.add_argument("--dataset_dir", default="dataset")
    args = parser.parse_args()

    evaluar_modelo(args.modelo, args.dataset_dir)
