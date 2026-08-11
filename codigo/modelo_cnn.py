"""
modelo_cnn.py  (versión DETECCIÓN DE OBJETOS)
------------------------------------------------
CNN construida desde cero, con DOS salidas (modelo multi-tarea):

  - clase_output: a qué etapa de madurez pertenece la planta (softmax, 3 clases)
  - caja_output:  dónde está la planta en la imagen (4 números: xmin,ymin,xmax,ymax
                   normalizados entre 0 y 1)

Ambas salidas comparten el mismo "tronco" convolucional (extracción
de características) y luego se separan en dos "cabezas" (heads)
independientes.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def crear_cnn(input_shape=(224, 224, 3), num_clases=3):
    # Definimos la entrada del modelo
    inputs = layers.Input(shape=input_shape)
    
    # --- Bloque Convolucional 1 ---
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # --- Bloque Convolucional 2 ---
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # --- Bloque Convolucional 3 ---
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    # --- Capas Densas Aplanadas ---
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x) # Dropout alto antes de las salidas para evitar memorización
    
    # --- SALIDAS DEL MODELO (Detección de Objetos Multi-salida) ---
    # Asegúrate de mantener estos nombres idénticos para que coincidan con entrenar.py
    clase_output = layers.Dense(num_clases, activation='softmax', name='clase_output')(x)
    caja_output = layers.Dense(4, activation='sigmoid', name='caja_output')(x)
    
    modelo = models.Model(inputs=inputs, outputs=[clase_output, caja_output])
    
    print("¡Estructura CNN optimizada contra sobreajuste creada!")
    return modelo
