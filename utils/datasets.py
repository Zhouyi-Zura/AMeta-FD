import glob, sys
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import skimage


class ImageDataset_Train(Dataset):
    def __init__(self, root, transforms_=None, task_flag=None, ML_mode=None):
        self.transform = transforms.Compose(transforms_)
        self.image_files = sorted(glob.glob(root + "/image/*.*"))
        self.label_files = sorted(glob.glob(root + "/label/*.*"))
        self.task_flag = task_flag
        self.ML_mode = ML_mode

    def add_noise(self, image, noise_mode):
        # -----------
        # Gau0.02 Gau0.03 Gau0.04 Gau0.05 Gau0.06 Gau0.07 Gau0.08 Gau0.09
        # GP0.02 GP0.03 GP0.04 GP0.05 GP0.06 GP0.07 GP0.08 GP0.09
        # -----------
        n_mode = noise_mode[:-4]
        Gau_var = noise_mode[-4:]
        if n_mode == "Gau":
            img = np.array(image)
            img = skimage.util.random_noise(img, mode='gaussian', var = float(Gau_var))
            noise_img = 255 * (img / np.amax(img))
        elif n_mode == "GP":
            img = np.array(image)
            img = skimage.util.random_noise(img, mode='gaussian', var = float(Gau_var))
            img = 255 * (img / np.amax(img))
            noise = np.random.poisson(img)
            noise_img = img + noise
            noise_img = 255 * (noise_img / np.amax(noise_img))
        
        noisy_PIL = Image.fromarray(noise_img)

        return noisy_PIL

    def __getitem__(self, index):
        img = Image.open(self.image_files[index % len(self.image_files)]).convert("L")
        lab = Image.open(self.label_files[index % len(self.label_files)]).convert("L")

        if (np.random.random() < 0.5):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lab = lab.transpose(Image.FLIP_LEFT_RIGHT)

        if self.ML_mode == 0:
            factor = np.random.randint(-15,15)
            img = img.rotate(factor)
            lab = lab.rotate(factor)
            img = self.transform(img)
            lab = self.transform(lab)
        elif self.ML_mode == 1:
            img = self.add_noise(lab, self.task_flag)
            factor = np.random.randint(-15,15)
            img = img.rotate(factor)
            lab = lab.rotate(factor)
            img = self.transform(img)
            lab = self.transform(lab)
        else:
            sys.exit("Wrong Meta-Learning mode!")

        return {"image": img, "label": lab}

    def __len__(self):
        return len(self.image_files)
