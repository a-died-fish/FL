import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_images2np(directory='gen'):
    image_list = []
    label_list = []

    # Iterate over each sub-directory
    for sub_dir in tqdm(os.listdir(directory), desc='Loading Generated Images'):
        sub_dir_path = os.path.join(directory, sub_dir)
        
        # Ensure it's a directory and the name is a number (to filter out any non-numeric directories)
        if os.path.isdir(sub_dir_path) and sub_dir.isdigit():
            label = int(sub_dir)  # The label is the directory name as an integer
            
            # Load each PNG image from this sub-directory
            for file_name in os.listdir(sub_dir_path):
                if file_name.endswith('.png'):
                    file_path = os.path.join(sub_dir_path, file_name)
                    
                    # Open the image with PIL and convert to NumPy array
                    with Image.open(file_path) as img:
                        img_array = np.asarray(img)
                        image_list.append(img_array)
                        label_list.append(label)

    # Convert lists to NumPy arrays
    images = np.array(image_list)
    labels = np.array(label_list)

    return images, labels


if __name__ == '__main__':
    path = '/GPFS/data/xinyuzhu/FedGC/data/vlcs/generation_SD-v1-5_domain_new/SUN'
    images, labels = load_images2np(path)
    print(images.shape)
    print(labels.shape)
    # save numpy
    np.save(os.path.join(path, 'images'), images)
    np.save(os.path.join(path, 'labels'), labels)
