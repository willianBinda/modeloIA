import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

#  configurações
data_dir = './dataset/lixo'
img_height, img_width = 384, 512
batch_size = 32
num_classes = 6
epochs = 25

# gerador de dados
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Criação dos geradores de treino e validação
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)


validation_generator = train_datagen.flow_from_directory(
    data_dir,
    # target_dir,
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

#modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
            #   loss='binary_crossentropy',
              metrics=['accuracy'])

# treinamento
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator)

model.save('modeloIA.h5')
print("Modelo treinado e salvo com sucesso!")


plt.plot(history.history['accuracy'], label='Acurácia de treino')
plt.plot(history.history['val_accuracy'], label='Acurácia de validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()