from flask import Flask, request, render_template, send_file
import torch
from torchvision.utils import make_grid
from PIL import Image
import torchvision.transforms as transforms
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from tools import sample_ddpm_class_cond as smp
from utils.config_utils import *
import numpy as np
import yaml
import os
from evaluation import eval
from evaluation import incep

app = Flask(__name__)

# Load your trained models and configurations here
# ... (code to load models and configurations)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

animal_mapping = {
            'dog': 0, 'cat': 1, 'horse': 2, 'spider': 3, 'butterfly': 4,
            'chicken': 5, 'sheep': 6, 'cow': 7, 'squirrel': 8, 'elephant': 9
        }

alpha_mapping = {
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
            'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
            'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25,
            '0': 26, '1': 27, '2': 28, '3': 29, '4': 30, '5': 31, '6': 32, '7': 33, '8': 34, '9': 35,
            '@': 36, '#': 37, '$': 38, '&': 39
        }

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config_path_alpha = '/mnt/nvme0n1p5/projects/hackathon/CP2/config/mnist_class_cond.yaml'
config_path_animals = '/mnt/nvme0n1p5/projects/hackathon/CP2/config/animal_class_cond.yaml'

config_alpha = load_config(config_path_alpha)
config_animals = load_config(config_path_animals)

# Load config alpha
diffusion_config_alpha = config_alpha['diffusion_params']
dataset_config_alpha = config_alpha['dataset_params']
diffusion_model_config_alpha = config_alpha['ldm_params']
autoencoder_model_config_alpha = config_alpha['autoencoder_params']
train_config_alpha = config_alpha['train_params']

# Load config animals
diffusion_config_animals = config_animals['diffusion_params']
dataset_config_animals = config_animals['dataset_params']
diffusion_model_config_animals = config_animals['ldm_params']
autoencoder_model_config_animals = config_animals['autoencoder_params']
train_config_animals = config_animals['train_params']

def load_models(train_config, diffusion_model_config, autoencoder_model_config, dataset_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load UNet model
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                      model_config=diffusion_model_config).to(device)
    model.eval()
    model.load_state_dict(torch.load(os.path.join('tools/mnist',train_config['ldm_ckpt_name']), map_location=device))

    # Load VQVAE model
    vae_model = VQVAE(im_channels=dataset_config['im_channels'],
                      model_config=autoencoder_model_config).to(device)
    vae_model.eval()
    vae_model.load_state_dict(torch.load(os.path.join('tools/mnist',train_config['vqvae_autoencoder_ckpt_name']), map_location=device))

    return model, vae_model

model_alpha, vae_model_alpha = load_models(train_config_alpha, diffusion_model_config_alpha, autoencoder_model_config_alpha, dataset_config_alpha)
model_animals, vae_model_animals = load_models(train_config_animals, diffusion_model_config_animals, autoencoder_model_config_animals, dataset_config_animals)

scheduler_alpha = LinearNoiseScheduler(num_timesteps=diffusion_config_alpha['num_timesteps'],
                                     beta_start=diffusion_config_alpha['beta_start'],
                                     beta_end=diffusion_config_alpha['beta_end'])
scheduler_animals = LinearNoiseScheduler(num_timesteps=diffusion_config_animals['num_timesteps'],
                                     beta_start=diffusion_config_animals['beta_start'],
                                     beta_end=diffusion_config_animals['beta_end'])
fid_score = 0
inception_score = 0
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    desired_class = (request.form['class'])
    print(desired_class)
    if desired_class in ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel']:
        with torch.no_grad():
            desired_class = animal_mapping.get(desired_class.lower())
            model_animals, vae_model_animals = load_models(train_config_animals, diffusion_model_config_animals,
                                                           autoencoder_model_config_animals, dataset_config_animals)
            sample_classes = torch.tensor([desired_class], dtype=torch.long).to(device)
            with torch.no_grad():
                generated_image_path = smp.sample_web(model_animals, scheduler_animals, train_config_animals, diffusion_model_config_animals,
                                                      autoencoder_model_config_animals, diffusion_config_animals, dataset_config_animals,
                                                      vae_model_animals, desired_class, 1)
        fid_score = eval.ret_fid_alpha()
        folder_path = '/mnt/nvme0n1p5/projects/hackathon/CP2/tools/mnist/cond_class_samples/2'
        images = incep.read_images_from_folder(folder_path)
        images_array = np.array(images)
        inception_score = round(incep.calculate_inception_score(images_array),2)

    else:
        with torch.no_grad():
            desired_class = alpha_mapping.get(desired_class)
            sample_classes = torch.tensor([desired_class], dtype=torch.long).to(device)

            with torch.no_grad():
                generated_image_path = smp.sample_web(model_alpha, scheduler_alpha, train_config_alpha, diffusion_model_config_alpha,
                   autoencoder_model_config_alpha, diffusion_config_alpha, dataset_config_alpha, vae_model_alpha, desired_class,1)

        fid_score = eval.ret_fid_alpha()
        folder_path = '/mnt/nvme0n1p5/projects/hackathon/CP2/tools/mnist/cond_class_samples/1'
        images = incep.read_images_from_folder(folder_path)
        images_array = np.array(images)
        inception_score = round(incep.calculate_inception_score(images_array),2)

        print(generated_image_path)
    return render_template('index.html', image_path=generated_image_path, fscore=fid_score, iscore=inception_score)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
