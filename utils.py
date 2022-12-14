import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from transformers import ViTFeatureExtractor



def save_ckpt(PATH, name_ckpt, model, optimizer, epoch):
    path_save = os.path.join(PATH, "epoch_{}".format(epoch))
    if not os.path.exists(path_save):
        os.mkdir(path_save)

    torch.save(model.state_dict(),os.path.join(path_save, name_ckpt))
    torch.save(optimizer.state_dict(),os.path.join(path_save, "optimizer.pt"))


class CashewDataset(Dataset):
    def __init__(self, dir_path,  data_transforms):
        self.dir_path = dir_path
        self.images = datasets.ImageFolder(
                        dir_path, transform=data_transforms
                    )
        self.label2id = self.images.class_to_idx

    
    def __getitem__(self, index):
        image, class_id = self.images[index] 
        return image, class_id
    
    def __len__(self):
        return len(self.images)

def get_data_transforms(model_name, tf_default):
    if tf_default:
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

        IMG_MEAN = feature_extractor.image_mean
        IMG_SDEV = feature_extractor.image_std
        IMG_SIZE = feature_extractor.size

        data_transforms = transforms.Compose([
                transforms.Resize((IMG_SIZE["height"], IMG_SIZE["width"])),
                transforms.ToTensor(),
                transforms.Normalize(IMG_MEAN, IMG_SDEV), 
        ])

    else:
        data_transforms = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
        ])

    return data_transforms

def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0]) 
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

