from PIL import Image

import torchvision.transforms as transforms
import torch

def make_transform(imsize):
    return transforms.Compose([transforms.Resize(imsize),transforms.ToTensor()])


def load_image(image_name,imsize,device='cuda'):
    image=Image.open(image_name)
    transform=make_transform(imsize)
    image=transform(image).unsqueeze(0)  # fake-batch 만들어줌
    return image.to(device, dtype=torch.float)