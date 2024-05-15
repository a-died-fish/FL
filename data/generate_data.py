from diffusers import StableDiffusionPipeline, DiffusionPipeline
import torch
import json
import os
import random
import numpy as np
from tqdm import tqdm
import sys
from PIL import Image

name2path = {
    'SD-v1-5': '/GPFS/data/yaxindu/multimodal/CLIP-diff/stable-diffusion-v1-5',
}

dataset2shape = {
    'cifar10': (32,32),
    'cifar100': (32,32),
    'eurosat':(64,64),
    'tiny_imagenet':(64,64),
    'pacs':(227,227),
    'officehome':(224,224),
    'vlcs':(227,227)
}

if __name__ == '__main__':
    torch.cuda.set_device(5)
    dataset_name = 'vlcs'
    dataset_dir = './' + dataset_name
    start_index = 1000    # set to number X if we already have generated X images for each category.
    n_gen_per_class = 1000
    batch_size =1
    gen_name = 'SD-v1-5'
    
    
    with open(os.path.join(dataset_dir, "class2name.json"), 'r') as file:
        class2name = json.load(file)
    gen_root = os.path.join(dataset_dir, f"generation_{gen_name}")
    os.makedirs(gen_root, exist_ok=True)

    model_id = name2path[gen_name]
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.safety_checker = lambda images, clip_input: (images, False)
    # model_id = "/GPFS/data/yaxindu/multimodal/CLIP-diff/Realistic_VIsion_V5.1_noVAE"
    # pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    for class_id, class_name in class2name.items():
        class_name = class_name.replace('_', ' ')
        print(f"{'='*20} Generate {class_id} : {class_name} {'='*20}")  
        gen_dir = os.path.join(gen_root, class_id)
        os.makedirs(gen_dir, exist_ok=True)
        template_list = [f"real {class_name} image", f"a real photo of a {class_name}",\
                     f"a real picture of a {class_name}", f"{class_name}"]            # TODO
        #template_list = [f"a satellite photo of a {class_name}",f"a satellite picture of a {class_name}"]
        for gen_id in tqdm(range(n_gen_per_class//batch_size)):
            prompt = random.choices(template_list, k=batch_size)
            guidance_scale = int(np.random.normal(10, 2, 1)[0])     # 7~12
            images = pipe(prompt, guidance_scale=guidance_scale).images
            for i, image in enumerate(images):
                resized_image = image.resize(dataset2shape[dataset_name], Image.LANCZOS)
                resized_image.save(os.path.join(gen_dir, f"{start_index+gen_id*batch_size+i}_{prompt[i].replace(' ', '-')}_{guidance_scale}.png"))
    
    print(f"{'='*20} Finish Generation {'='*20}")
    print(f"{'='*20} Now you can move to png2np.py! {'='*20}")