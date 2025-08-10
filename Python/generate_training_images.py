import pandas as pd
import numpy as np
from PIL import Image
import os

os.makedirs('/home/carrotcake/documents/projects/personal/BasicNeuralNetwork/assets/mnist_images', exist_ok=True)

df = pd.read_csv('/home/carrotcake/documents/projects/personal/BasicNeuralNetwork/assets/archive/mnist_train.csv')

for i, row in df.iterrows():
    label = row[0]
    pixels = row[1:].values.astype(np.uint8).reshape(28, 28)
    img = Image.fromarray(pixels, mode='L')  # 'L' = grayscale
    img.save(f'/home/carrotcake/documents/projects/personal/BasicNeuralNetwork/assets/mnist_images/{i}_{label}.png')

    if i >= 99:  # just convert first 100 images for now
        break
