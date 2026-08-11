"""
entrenar.py
-----------
Entrena el modelo indicado (CNN desde cero o MobileNetV2) para DETECCIÓN DE OBJETOS y guarda:
  - el modelo entrenado (.keras)
  - las curvas de accuracy (clasificación) y loss (total) en resultados/
  - el historial de entrenamiento en formato .json

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

    # En detección multi-salida, Keras nombra las métricas combinando
    # el nombre de la capa de salida. Buscamos las llaves dinámicamente.
    keys = historial.history.keys()
    acc_key = [k for k in keys if "accuracy" in k and "val" not in k][0]
    val_acc_key = [k for k in keys if "accuracy" in k and "val" in k][0]

    # --- Accuracy de Clasificación ---
    plt.figure()
    plt.plot(historial.history[acc_key], label="Entrenamiento")
    plt.plot(historial.history[val_acc_key], label="Validación")
    plt.title(f"Clasificación Accuracy - {nombre_modelo}")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{carpeta_salida}/accuracy_{nombre_modelo}.png", bbox_inches="tight")
    plt.close()

    # --- Loss Total (Clasificación + Caja) ---
    plt.figure()
    plt.plot(historial.history["loss"], label="Entrenamiento")
    plt.plot(historial.history["val_loss"], label="Validación")
    plt.title(f"Loss Total - {nombre_modelo}")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{carpeta_salida}/loss_{nombre_modelo}.png", bbox_inches="tight")
    plt.close()

    with open(f"{carpeta_salida}/historial_{nombre_modelo}.json", "w") as f:
        json.dump(historial.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Entrena la CNN o el modelo MobileNetV2 para Detección.")
    parser.add_argument("--modelo", choices=["cnn", "mobilenet"], required=True,
                         help="Qué arquitectura entrenar.")
    parser.add_argument("--dataset_dir", default="dataset",
                         help="Carpeta con train/valid/test (exportada desde Roboflow en formato CSV).")
    parser.add_argument("--epocas", type=int, default=20)
    parser.add_argument("--fine_tune", action="store_true",
                         help="Solo aplica a --modelo mobilenet: habilita fine-tuning.")
    args = parser.parse_args()

    # cargar_datasets ahora devuelve, para cada elemento del dataset:
    # (imagen, {"clase_output": etiqueta, "caja_output": [xmin,ymin,xmax,ymax]})
    train_ds, val_ds, test_ds = cargar_datasets(args.dataset_dir)

    if args.modelo == "cnn":
        modelo = crear_cnn()
        nombre = "cnn"
    else:
        modelo = crear_mobilenet(fine_tune=args.fine_tune)
        nombre = "mobilenet"

    # CONFIGURACIÓN PARA DETECCIÓN DE OBJETOS (Multi-salida)
    # Se usan pérdidas independientes para la categoría de madurez y la regresión de la caja bounding box.
    modelo.compile(
        optimizer="adam",
        loss={
            "clase_output": "sparse_categorical_crossentropy",  # Madurez (0, 1, 2)
            "caja_output": "mean_squared_error"                  # Coordenadas [xmin, ymin, xmax, ymax]
        },
        loss_weights={
            "clase_output": 1.0,
            "caja_output": 1.0
        },
        metrics={
            "clase_output": ["accuracy"],
            "caja_output": ["mae"]  # Error absoluto medio para evaluar la precisión del cuadro
        }
    )

    # =================================================================
    # NUEVO: ESTRATEGIAS DE GENERALIZACIÓN (Para el informe del profesor)
    # =================================================================

    # 1. EarlyStopping evita seguir entrenando si la pérdida de validación deja de mejorar.
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    
    # 2. Learning Rate Scheduler (Opcional según rúbrica)
    # Reduce la velocidad de aprendizaje 10% cada época después de la época 10
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            import math
            return float(lr * math.exp(-0.1))
           
            
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # Juntamos ambas estrategias
    callbacks = [early_stop, lr_scheduler]

    # =================================================================

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

    # Evaluación rápida sobre el set de prueba al final.
    resultados = modelo.evaluate(test_ds)
    print("\nResultado en test:")
    for nombre_metrica, valor in zip(modelo.metrics_names, resultados):
        print(f"  {nombre_metrica}: {valor:.4f}")


if __name__ == "__main__":
    main()
