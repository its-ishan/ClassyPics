import glob
import os
import torchvision
from PIL import Image
from tqdm import tqdm
from utils.diffusion_utils import load_latents
from torch.utils.data.dataset import Dataset
import cv2
import torch
from tools import translate


class AnimalDataset(Dataset):
    
    def __init__(self, split, im_path, im_size, im_channels,
                 use_latents=False, latent_path=None, condition_config=None):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels

        # Should we use latents or not
        self.latent_maps = None
        self.use_latents = False
        
        # Conditioning for the dataset
        self.condition_types = [] if condition_config is None else condition_config['condition_types']

        self.images, self.labels = self.load_images(im_path)
        #self.class_mapping = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.class_mapping = {
            'dog': 0, 'cat': 1, 'horse': 2, 'spider': 3, 'butterfly': 4,
            'chicken': 5, 'sheep': 6, 'cow': 7, 'squirrel': 8, 'elephant': 9
        }

        # Whether to load images and call vae or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')
        
    def load_images(self, im_path):
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            fnames = glob.glob(os.path.join(im_path, d_name, '*.{}'.format('png')))
            fnames += glob.glob(os.path.join(im_path, d_name, '*.{}'.format('jpg')))
            fnames += glob.glob(os.path.join(im_path, d_name, '*.{}'.format('jpeg')))
            for fname in fnames:
                ims.append(fname)
                if 'class' in self.condition_types:
                    labels.append(translate.get(d_name))
                    #labels.append(int(d_name))
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        cond_inputs = {}
        # if 'class' in self.condition_types:
        #     cond_inputs['class'] = self.labels[index]
        if 'class' in self.condition_types:
            # Map class string to integer
            mapped_class = self.class_mapping[self.labels[index]]
            cond_inputs['class'] = mapped_class

        #######################################
        
        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs
        else:
            #im = Image.open(self.images[index])
            im = cv2.imread(self.images[index])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            # resize_transform = torchvision.transforms.Resize((32, 32))
            # im = resize_transform(im)

            im = cv2.resize(im, (64, 64))

            #im_tensor = torchvision.transforms.ToTensor()(im)
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)

            # Convert input to -1 to 1 range.
            #im_tensor = (2 * im_tensor) - 1
            im_tensor = (im_tensor / 255.0) * 2 - 1
            if len(self.condition_types) == 0:
                return im_tensor
            else:
                return im_tensor, cond_inputs
            

if __name__ == '__main__':
    dataset = AnimalDataset('train', '/mnt/nvme0n1p5/projects/hackathon/CP2/data/Animals10/raw-img/', 28, 1)

    first_item = dataset[0]
    print(first_item)