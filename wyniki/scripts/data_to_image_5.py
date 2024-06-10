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
                tempData = df.values[:, 1:]
                num_of_markers = tempData.shape[1] // 3
                reshaped_data = np.zeros_like(tempData, dtype=np.uint8)
                for i in range(num_of_markers*3):
                    channel_data = tempData[:, i]
                    if channel_data.size > 0:  # Sprawdzenie, czy channel_data nie jest pusty
                        min_value = np.min(channel_data)
                        max_value = np.max(channel_data)
                        reshaped_data[:, i] = rescale_to_255(channel_data, min_value, max_value)
                
                reshaped_data = reshaped_data.reshape(40, num_of_markers, 3)
                reshaped_data = reshaped_data.transpose(1, 0, 2)

                image = Image.fromarray(reshaped_data, 'RGB')
                image = image.resize((224, 224), Image.NEAREST)

                file_basename = Path(filename).stem

                if label == 0:
                    output_path = os.path.join(images_folder+"walk_comfortable", f'{file_basename}{image_extension}')
                elif label == 1:
                    output_path = os.path.join(images_folder+"walk_fast", f'{file_basename}{image_extension}')
                elif label == 2:
                    output_path = os.path.join(images_folder+"walk_slow", f'{file_basename}{image_extension}')
                elif label == 3:
                    output_path = os.path.join(images_folder+"run_comfortable", f'{file_basename}{image_extension}')
                elif label == 4:
                    output_path = os.path.join(images_folder+"run_fast", f'{file_basename}{image_extension}')
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Tworzenie folderu, jeśli nie istnieje
                image.save(output_path)

                # Dodanie obrazu do danych (data) i etykiet (labels)
                data.append(np.array(image))
                labels.append(folder)
    return np.array(data), np.array(labels)

#main_folder = '../data_5_classes/'
main_folder = '../data_5_classes_4_markers/'
#images_folder = '../images_5/'
images_folder = '../images_5_4_markers/'
image_extension = '.jpeg'
data, labels = load_data(main_folder)