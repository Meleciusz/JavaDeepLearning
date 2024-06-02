import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# Ścieżki do folderów z obrazami

################ 2 klasy
#train_dir = '../images_2'
#class_mode_arg = "binary"
#model_path = "../model/img_model_2.h5"

################ 5 klasy 20 markery
#train_dir = '../images_5'
#class_mode_arg = "categorical"
#model_path = "../model/img_model_5.h5"

################ 5 klasy 4 markery
train_dir = '../images_5_4_markers'
class_mode_arg = "categorical"
model_path = "../model/img_model_5_4_markers.h5"

# Parametry przetwarzania obrazów
img_height, img_width = 224, 224  # Rozmiar oczekiwany przez VGG16
batch_size = 32

# Przygotowanie generatorów danych z augmentacją dla zwiększenia różnorodności danych treningowych
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.33,  # Podział na dane treningowe i walidacyjne
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    shear_range=0.0,
    zoom_range=0.0,
    horizontal_flip=False,
    fill_mode='nearest'
)

# Generator dla danych treningowych
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=class_mode_arg,
    subset='training'  # Podzbiór treningowy
)

# Generator dla danych walidacyjnych
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=class_mode_arg,
    subset='validation'  # Podzbiór walidacyjny
)

# Załadowanie modelu VGG16 bez górnych warstw
#base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

### Zablokowanie trenowania wcześniej nauczonych warstw
#for layer in base_model.layers:
#    layer.trainable = False
################# Wersja z VGG16
## Dodanie własnych warstw na górze
#model = models.Sequential([
#    base_model,
#    layers.Flatten(),
#    layers.Dense(256, activation='relu'),
#    layers.Dropout(0.5),
#    layers.Dense(1, activation='sigmoid')
#])
################ Wersja bez VGG16

model = models.Sequential([
    layers.Flatten(input_shape=(img_height, img_width, 3)),  # Warstwa spłaszczająca obrazy do wektora cech
    layers.Dense(256, activation='relu'),  # Warstwa gęsta z 256 neuronami i funkcją aktywacji ReLU
    layers.Dropout(0.5),  # Warstwa dropout do regularyzacji
    layers.Dense(1, activation='sigmoid')  # Warstwa wyjściowa z jednym neuronem i funkcją aktywacji sigmoid
])

# Kompilacja modelu
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Trening modelu
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Zapis modelu
model.save(model_path)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='upper right')

plt.show()

print("Accuracy on training data:", history.history['accuracy'])
print("Accuracy on validation data:", history.history['val_accuracy'])