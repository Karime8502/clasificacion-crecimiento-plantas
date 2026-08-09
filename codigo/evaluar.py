"""
evaluar.py
----------
Evalúa un modelo ya entrenado (.keras) sobre el conjunto de prueba y
calcula las métricas exigidas por el profesor para tareas de
clasificación: Accuracy, Precision, Recall, F1-Score, ROC-AUC y
matriz de confusión.

Uso:
    python evaluar.py --modelo modelos/modelo_mobilenet.keras --dataset_dir dataset
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

from preparar_datos import cargar_datasets


def evaluar_modelo(ruta_modelo, dataset_dir="dataset", carpeta_salida="resultados"):
    os.makedirs(carpeta_salida, exist_ok=True)

    _, _, test_ds = cargar_datasets(dataset_dir)
    nombres_clases = test_ds.class_names

    modelo = tf.keras.models.load_model(ruta_modelo)

    y_true = []
    y_pred_probs = []

    for imagenes, etiquetas in test_ds:
        probs = modelo.predict(imagenes, verbose=0)
        y_pred_probs.append(probs)
        y_true.append(etiquetas.numpy())

    y_true = np.concatenate(y_true)
    y_pred_probs = np.concatenate(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n=== Reporte de clasificación (accuracy, precision, recall, F1) ===")
    reporte = classification_report(y_true, y_pred, target_names=nombres_clases)
    print(reporte)

    # ROC-AUC multiclase (one-vs-rest), útil cuando las clases no son binarias.
    try:
        auc = roc_auc_score(y_true, y_pred_probs, multi_class="ovr")
        print(f"ROC-AUC (one-vs-rest): {auc:.4f}")
    except ValueError as e:
        auc = None
        print(f"No se pudo calcular ROC-AUC: {e}")

    # --- Matriz de confusión ---
    matriz = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=nombres_clases)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Matriz de confusión")
    ruta_matriz = f"{carpeta_salida}/matriz_confusion.png"
    plt.savefig(ruta_matriz, bbox_inches="tight")
    plt.close()
    print(f"\nMatriz de confusión guardada en {ruta_matriz}")

    # Guardar el reporte de texto también, para anexarlo al informe.
    with open(f"{carpeta_salida}/reporte_clasificacion.txt", "w") as f:
        f.write(reporte)
        if auc is not None:
            f.write(f"\nROC-AUC (one-vs-rest): {auc:.4f}\n")

    return reporte, matriz, auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa un modelo entrenado.")
    parser.add_argument("--modelo", required=True, help="Ruta al archivo .keras")
    parser.add_argument("--dataset_dir", default="dataset")
    args = parser.parse_args()

    evaluar_modelo(args.modelo, args.dataset_dir)
