import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

def rescale_to_255(data, min_val, max_val):
    data_rescaled = (data - min_val) / (max_val - min_val) * 255
    return data_rescaled.astype(np.uint8)

def load_data(folder):
    data = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder, filename)
            df = pd.read_csv(filepath)
            tempData = df.values[:, 1:]
            if tempData.shape == (40, 12):
                reshaped_data = np.zeros_like(tempData, dtype=np.uint8)
                for i in range(12):
                    channel_data = tempData[:, i]
                    if channel_data.size > 0:  # Sprawdzenie, czy channel_data nie jest pusty
                        min_value = np.min(channel_data)
                        max_value = np.max(channel_data)
                        reshaped_data[:, i] = rescale_to_255(channel_data, min_value, max_value)
                
                reshaped_data = reshaped_data.reshape(40, 4, 3)
                reshaped_data = reshaped_data.transpose(1, 0, 2)

                image = Image.fromarray(reshaped_data, 'RGB')
                image = image.resize((224, 224), Image.NEAREST)

                file_basename = Path(filename).stem

                if folder == run_folder:
                    output_path = os.path.join(run_images_folder, f'{file_basename}{image_extension}')
                else:
                    output_path = os.path.join(walk_images_folder, f'{file_basename}{image_extension}')
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Tworzenie folderu, jeśli nie istnieje
                image.save(output_path)

                # Dodanie obrazu do danych (data) i etykiet (labels)
                data.append(np.array(image))
                labels.append(folder)  # Możesz tutaj użyć innych etykiet, np. 'run' lub 'walk'

    return np.array(data), np.array(labels)

run_folder = '../data_2_classes/run'
run_images_folder = '../images_2/run'

walk_folder = '../data_2_classes/walk'
walk_images_folder = '../images_2/walk'

image_extension = '.jpeg'
image_scale = 15

load_data(run_folder)
load_data(walk_folder)
