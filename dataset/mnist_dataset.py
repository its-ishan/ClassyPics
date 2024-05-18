import glob
import os
import torchvision
from PIL import Image
from tqdm import tqdm
from utils.diffusion_utils import load_latents
from torch.utils.data.dataset import Dataset


class MnistDataset(Dataset):
    r"""
    Nothing special here. Just a simple dataset class for mnist images.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset
    """
    
    def __init__(self, split, im_path, im_size, im_channels,
                 use_latents=False, latent_path=None, condition_config=None):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
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
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
            'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
            'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25,
            '0': 26, '1': 27, '2': 28, '3': 29, '4': 30, '5': 31, '6': 32, '7': 33, '8': 34, '9': 35,
            '@': 36, '#': 37, '$': 38, '&': 39
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
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path:
        :return:
        """
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
                    labels.append(d_name)
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
            im = Image.open(self.images[index])
            resize_transform = torchvision.transforms.Resize((32, 32))
            im = resize_transform(im)

            im_tensor = torchvision.transforms.ToTensor()(im)

            # Convert input to -1 to 1 range.
            im_tensor = (2 * im_tensor) - 1
            if len(self.condition_types) == 0:
                return im_tensor
            else:
                return im_tensor, cond_inputs
            

if __name__ == '__main__':
    dataset = MnistDataset('train', '/mnt/nvme0n1p5/projects/hackathon/CP2/data/mnist/train/images/', 28, 1)
    print(len(dataset))
    print(dataset[0])
    print(dataset[0][0].shape)
    #print(dataset[0][1])