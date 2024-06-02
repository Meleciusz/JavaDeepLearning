import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16

# Funkcja do wczytywania danych z folderów
def load_data(folder):
    data = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder, filename)
            df = pd.read_csv(filepath)
            data.append(df.values[:, 1:])  # pomijamy kolumnę czasu
            if 'Run' in filename:
                labels.append(1)  # bieg
            else:
                labels.append(0)  # chód
    return np.array(data), np.array(labels)

# Wczytanie danych
run_folder = '../data_2_classes/run'  # Zmień na właściwą ścieżkę
walk_folder = '../data_2_classes/walk'  # Zmień na właściwą ścieżkę

run_data, run_labels = load_data(run_folder)
walk_data, walk_labels = load_data(walk_folder)

# Łączenie i przygotowanie danych
X = np.concatenate((run_data, walk_data), axis=0)
y = np.concatenate((run_labels, walk_labels), axis=0)

# Normalizacja danych
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Podział danych na treningowe (2/3) i walidacyjne (1/3)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

# Zakładamy, że dane mają 13 kolumn (3 osie dla 4 punktów ciała i czas)
# Sprawdzamy wymiary danych
n_samples, n_timesteps, n_features = X_train.shape

# Musimy upewnić się, że liczba cech (n_features) jest podzielna przez 3
if n_features % 3 != 0:
    raise ValueError("Liczba cech nie jest podzielna przez liczbę osi (3). Sprawdź dane wejściowe.")

n_points = n_features // 3  # liczba punktów ciała 

# Zmieniamy kształt na (n_samples, n_timesteps, n_features) -> (n_samples, n_timesteps, n_points, 3)
X_train = X_train.reshape((X_train.shape[0], n_timesteps, n_points* 3))
X_val = X_val.reshape((X_val.shape[0], n_timesteps, n_points* 3))

# Konwersja etykiet na one-hot encoding
y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)

# Definicja modelu CNN
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(n_timesteps, n_points * 3)),
    MaxPooling1D(2),
    Dropout(0.5),
    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
model.fit(X_train, y_train, epochs=50, batch_size=160, validation_data=(X_val, y_val))

# Ewaluacja modelu
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation accuracy: {accuracy}')

# Zapisanie modelu
model.save('../model/data_model_2.h5')

