import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

################ 2 klasy
train_dir = '../images_2'
class_mode_arg = "binary"
model_path = "../model/img_model_2.h5"

img_height, img_width = 224, 224 
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.33
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=class_mode_arg,
    subset='training'  
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=class_mode_arg,
    subset='validation'  
)

model = models.Sequential([
    layers.Flatten(input_shape=(img_height, img_width, 3)),  # Warstwa spłaszczająca obrazy do wektora cech
    layers.Dense(256, activation='relu'),  # Warstwa gęsta z 256 neuronami i funkcją aktywacji ReLU
    layers.Dropout(0.5),  # Warstwa dropout do regularyzacji
    layers.Dense(1, activation='sigmoid')  # Warstwa wyjściowa z jednym neuronem i funkcją aktywacji sigmoid
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=32,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

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