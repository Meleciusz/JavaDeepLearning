import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# Funkcja do wczytywania danych z folderów
def load_data(main_folder):
    data = []
    labels = []
    label_map = {
        'walk/Comfortable': 0, 
        'walk/Fast': 1, 
        'walk/Slow': 2,
        'run/Comfortable': 3, 
        'run/Fast': 4
    }
    
    for sub_folder, label in label_map.items():
        full_path = os.path.join(main_folder, sub_folder)
        for filename in os.listdir(full_path):
            if filename.endswith(".csv"):
                filepath = os.path.join(full_path, filename)
                df = pd.read_csv(filepath)
                data.append(df.values[:, 1:])  # pomijamy kolumnę czasu
                labels.append(label)
                
    return np.array(data), np.array(labels)

# Wczytanie danych
main_folder = '../data_5_classes/'  # Zmień na właściwą ścieżkę, jeśli jest inna
#main_folder = '../data_5_classes_4_markers/'

data, labels = load_data(main_folder)

# Łączenie i przygotowanie danych
X = data
y = labels

# Normalizacja danych
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Spłaszczanie danych do formatu 2D (n_samples, n_timesteps * n_features)
n_samples, n_timesteps, n_features = X.shape
X = X.reshape((n_samples, n_timesteps * n_features))

# Podział danych na treningowe (2/3) i walidacyjne (1/3)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

# Konwersja etykiet na one-hot encoding
y_train = to_categorical(y_train, num_classes=5)
y_val = to_categorical(y_val, num_classes=5)

# Definicja modelu sieci gęstej z BatchNormalization
model = Sequential([
    Dense(256, activation='relu', input_shape=(n_timesteps * n_features,)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 klas
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Ewaluacja modelu
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation accuracy: {accuracy}')

# Zapisanie modelu
model.save('../model/data_model_5.h5')

#import os
#import numpy as np
#import pandas as pd
#from sklearn.model_selection import train_test_split
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
#from tensorflow.keras.utils import to_categorical

## Funkcja do wczytywania danych z folderów
#def load_data(main_folder):
#    data = []
#    labels = []
#    label_map = {
#        'walk/Comfortable': 0, 
#        'walk/Fast': 1, 
#        'walk/Slow': 2,
#        'run/Comfortable': 3, 
#        'run/Fast': 4
#    }
    
#    for sub_folder, label in label_map.items():
#        full_path = os.path.join(main_folder, sub_folder)
#        for filename in os.listdir(full_path):
#            if filename.endswith(".csv"):
#                filepath = os.path.join(full_path, filename)
#                df = pd.read_csv(filepath)
#                data.append(df.values[:, 1:])  # pomijamy kolumnę czasu
#                labels.append(label)
                
#    return np.array(data), np.array(labels)

## Wczytanie danych
#main_folder = '../wyniki_other_markers_2/'  # Zmień na właściwą ścieżkę

#data, labels = load_data(main_folder)

## Łączenie i przygotowanie danych
#X = data
#y = labels

## Normalizacja danych
#X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

## Podział danych na treningowe (2/3) i walidacyjne (1/3)
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

## Zakładamy, że dane mają 12 kolumn (3 osie dla 4 punktów ciała)
## Sprawdzamy wymiary danych
#n_samples, n_timesteps, n_features = X_train.shape

## Musimy upewnić się, że liczba cech (n_features) jest podzielna przez 3
#if n_features % 3 != 0:
#    raise ValueError("Liczba cech nie jest podzielna przez liczbę osi (3). Sprawdź dane wejściowe.")

#n_points = n_features // 3  # liczba punktów ciała 

## Zmieniamy kształt na (n_samples, n_timesteps, n_features) -> (n_samples, n_timesteps, n_points, 3)
#X_train = X_train.reshape((X_train.shape[0], n_timesteps, n_points* 3))
#X_val = X_val.reshape((X_val.shape[0], n_timesteps, n_points* 3))

## Konwersja etykiet na one-hot encoding
#y_train = to_categorical(y_train, num_classes=5)
#y_val = to_categorical(y_val, num_classes=5)

## Definicja modelu CNN z BatchNormalization
#model = Sequential([
#    Conv1D(64, 3, activation='relu', input_shape=(n_timesteps, n_points * 3)),
#    BatchNormalization(),
#    MaxPooling1D(2),
#    Dropout(0.5),
#    Conv1D(128, 3, activation='relu'),
#    BatchNormalization(),
#    MaxPooling1D(2),
#    Dropout(0.5),
#    Flatten(),
#    Dense(128, activation='relu'),
#    Dropout(0.5),
#    Dense(5, activation='softmax')  # 5 klas
#])

## Kompilacja modelu
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

## Trenowanie modelu
#history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val))

## Ewaluacja modelu
#loss, accuracy = model.evaluate(X_val, y_val)
#print(f'Validation accuracy: {accuracy}')

## Zapisanie modelu
#model.save('gait_recognition_model.h5')
