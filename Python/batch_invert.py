from PIL import Image, ImageOps
import os

input_folder = '/home/carrotcake/documents/projects/personal/BasicNeuralNetwork/assets/' 
output_folder = '/home/carrotcake/documents/projects/personal/BasicNeuralNetwork/assets/inverted/'

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        print(f"Processing {filename}...")
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert('L')  # convert to grayscale
        
        inverted_img = ImageOps.invert(img)
        
        inverted_img.save(os.path.join(output_folder, filename))
        print(f"Inverted and saved {filename}")
