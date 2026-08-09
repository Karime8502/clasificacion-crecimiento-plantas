"""
entrenar.py
-----------
Entrena el modelo indicado (CNN desde cero o MobileNetV2) y guarda:
  - el modelo entrenado (.keras)
  - las curvas de accuracy y loss (resultados/accuracy.png, loss.png)
  - el historial de entrenamiento en formato .json (para el informe)

Uso:
    python entrenar.py --modelo cnn
    python entrenar.py --modelo mobilenet
    python entrenar.py --modelo mobilenet --epocas 30 --dataset_dir dataset
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf

from preparar_datos import cargar_datasets
from modelo_cnn import crear_cnn
from modelo_mobilenet import crear_mobilenet


def graficar_historial(historial, nombre_modelo, carpeta_salida="resultados"):
    os.makedirs(carpeta_salida, exist_ok=True)

    # --- Accuracy ---
    plt.figure()
    plt.plot(historial.history["accuracy"], label="Entrenamiento")
    plt.plot(historial.history["val_accuracy"], label="Validación")
    plt.title(f"Accuracy - {nombre_modelo}")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{carpeta_salida}/accuracy_{nombre_modelo}.png", bbox_inches="tight")
    plt.close()

    # --- Loss ---
    plt.figure()
    plt.plot(historial.history["loss"], label="Entrenamiento")
    plt.plot(historial.history["val_loss"], label="Validación")
    plt.title(f"Loss - {nombre_modelo}")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{carpeta_salida}/loss_{nombre_modelo}.png", bbox_inches="tight")
    plt.close()

    with open(f"{carpeta_salida}/historial_{nombre_modelo}.json", "w") as f:
        json.dump(historial.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Entrena la CNN o el modelo MobileNetV2.")
    parser.add_argument("--modelo", choices=["cnn", "mobilenet"], required=True,
                         help="Qué arquitectura entrenar.")
    parser.add_argument("--dataset_dir", default="dataset",
                         help="Carpeta con train/valid/test (exportada desde Roboflow).")
    parser.add_argument("--epocas", type=int, default=20)
    parser.add_argument("--fine_tune", action="store_true",
                         help="Solo aplica a --modelo mobilenet: habilita fine-tuning.")
    args = parser.parse_args()

    train_ds, val_ds, test_ds = cargar_datasets(args.dataset_dir)

    if args.modelo == "cnn":
        modelo = crear_cnn()
        nombre = "cnn"
    else:
        modelo = crear_mobilenet(fine_tune=args.fine_tune)
        nombre = "mobilenet"

    # sparse_categorical_crossentropy porque las etiquetas vienen como
    # enteros (0, 1, 2) y no como vectores one-hot.
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # EarlyStopping evita seguir entrenando (y sobreajustando) una vez
    # que la pérdida de validación deja de mejorar.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
    ]

    historial = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epocas,
        callbacks=callbacks,
    )

    os.makedirs("modelos", exist_ok=True)
    ruta_modelo = f"modelos/modelo_{nombre}.keras"
    modelo.save(ruta_modelo)
    print(f"Modelo guardado en {ruta_modelo}")

    graficar_historial(historial, nombre)

    # Evaluación rápida sobre el set de prueba, al final del entrenamiento.
    test_loss, test_acc = modelo.evaluate(test_ds)
    print(f"\nResultado en test -> loss: {test_loss:.4f}  accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
