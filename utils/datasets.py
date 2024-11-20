import glob, sys
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
from PIL import Image


class ImageDataset_Train(Dataset):
    def __init__(self, root, transforms_=None, task_flag=None, ML_mode=None):
        self.transform = transforms.Compose(transforms_)
        self.image_files = sorted(glob.glob(root + "/image/*.*"))
        self.label_files = sorted(glob.glob(root + "/label/*.*"))
        self.task_flag = task_flag
        self.ML_mode = ML_mode

    def add_noise(self, image, noise_mode):
        # -----------
        # Ray20 Ray30 Ray40 Ray50 Ray60 Ray70 Ray80 Ray90
        # RP20 RP30 RP40 RP50 RP60 RP70 RP80 RP90
        # -----------
        n_mode = noise_mode[:-2]
        Ray_sigma = noise_mode[-2:]
        image = image.astype(np.float32) / 255.0
        if n_mode == "Ray":
            rayleigh_noise1 = np.random.rayleigh(scale=0.1, size=image.shape)
            rayleigh_noise2 = np.random.rayleigh(scale=(Ray_sigma/100), size=image.shape)
            noise_img = (image + rayleigh_noise1) * rayleigh_noise2
            noise_img = np.clip(noise_img * 255, 0, 255).astype(np.uint8)
        elif n_mode == "RP":
            rayleigh_noise1 = np.random.rayleigh(scale=0.1, size=image.shape)
            rayleigh_noise2 = np.random.rayleigh(scale=(Ray_sigma/100), size=image.shape)
            noise_p = np.random.poisson(image)
            noise_img = (image + rayleigh_noise1 + noise_p) * rayleigh_noise2
            noise_img = np.clip(noise_img * 255, 0, 255).astype(np.uint8)
        
        noisy_PIL = Image.fromarray(noise_img)

        return noisy_PIL

    def __getitem__(self, index):
        img = cv2.imread(self.image_files[index % len(self.image_files)], cv2.IMREAD_GRAYSCALE)
        lab = cv2.imread(self.label_files[index % len(self.label_files)], cv2.IMREAD_GRAYSCALE)

        if self.ML_mode == 1:
            img = self.add_noise(lab, self.task_flag)
            img = self.transform(img)
            lab = self.transform(Image.fromarray(lab))
        elif self.ML_mode == 0:
            img = self.transform(Image.fromarray(img))
            lab = self.transform(Image.fromarray(lab))
        else:
            sys.exit("Wrong Meta-Learning mode!")

        return {"image": img, "label": lab}

    def __len__(self):
        return len(self.image_files)
