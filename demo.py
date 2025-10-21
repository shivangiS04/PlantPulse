import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Load sensor data CSV
sensor_df = pd.read_csv('sensor_data.csv')

# Base folder for PlantVillage images
base_folder = 'data/PlantVillage'

# Iterate over sensor data
for _, row in sensor_df.iterrows():
    image_path = os.path.join(base_folder, row['image'])
    if os.path.exists(image_path):
        image = Image.open(image_path)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Moisture: {row['sensor_moisture']}% | Temp: {row['temperature']}°C")
        plt.show()

        # Simple rule-based alert
        if row['sensor_moisture'] < 30 < row['temperature']:
            print("ALERT: Possible crop stress detected!")
        else:
            print("Status: Crop conditions normal.")
    else:
        print(f"{image_path} not found")
