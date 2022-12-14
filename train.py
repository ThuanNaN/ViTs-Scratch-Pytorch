import torch 
import time
import copy
import logging
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from yaml.loader import SafeLoader

from utils import *
from models.vit import ViT



logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger("Torch-Cls")


def train_model(model, dataloaders, criterion, optimizer, opt):
    device, num_epochs, PATH_SAVE, num_cls = opt.device, opt.n_epochs, opt.PATH_SAVE, opt.n_classes

    since = time.time()
    LOGGER.info(f"\n{colorstr('Optimizer:')} {optimizer}")
    LOGGER.info(f"\n{colorstr('Loss:')} {type(criterion).__name__}")

    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_confusion_matrix = torch.zeros(num_cls, num_cls)

    model.to(device)
    for epoch in range(num_epochs):
        LOGGER.info(colorstr(f'\nEpoch {epoch}/{num_epochs-1}:'))

        for phase in ["train", "val"]:
            if phase == "train":
                LOGGER.info(colorstr('bright_yellow', 'bold', '\n%20s' + '%15s' * 3) % 
                                ('Training:', 'gpu_mem', 'loss', 'acc'))
                model.train()
            else:
                LOGGER.info(colorstr('bright_yellow', 'bold', '\n%20s' + '%15s' * 3) % 
                                ('Validation:','gpu_mem', 'loss', 'acc'))
                model.eval()

            running_items = 0
            running_loss = 0.0
            running_corrects = 0
            confusion_matrix = torch.zeros(num_cls, num_cls)

            with tqdm(dataloaders[phase],
                total=len(dataloaders[phase]),
                bar_format='{desc} {percentage:>7.0f}%|{bar:10}{r_bar}{bar:-10b}',
                unit='batch') as _phase:

                for inputs, labels in _phase:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_items += inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    for t, p in zip(labels.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

                    epoch_loss = running_loss / running_items
                    epoch_acc = running_corrects.float() / running_items

                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}GB'
                    desc = ('%35s' + '%15.6g' * 2) % (mem, epoch_loss, epoch_acc)
                    _phase.set_description_str(desc)


            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_ckpt(PATH_SAVE, "best.pt", model, optimizer, epoch)
                best_confusion_matrix = confusion_matrix
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s with {} epochs'.format(time_elapsed // 60, time_elapsed % 60, num_epochs))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    # save_ckpt(PATH_SAVE, "last.pt", model, optimizer, epoch, loss)

    return model, val_acc_history, best_confusion_matrix




if __name__ == "__main__":


    with open("./config/data_config.yaml", "r") as f:
        DATA_CONFIG  = yaml.load(f, Loader=SafeLoader)


    with open("./config/train_config.yaml") as f:
        opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))

    opt.n_classes = DATA_CONFIG["n_classes"]
    # opt.device = "cuda" if torch.cuda.is_available() else "cpu"
    opt.device = torch.device("mps")

    data_transforms = get_data_transforms(opt.model_name, opt.default_data_transform)

    print("Data transforms: ")
    print(data_transforms)

    train_dataset = CashewDataset(DATA_CONFIG["train"], data_transforms)
    val_dataset = CashewDataset(DATA_CONFIG["val"], data_transforms)

    label_dict = train_dataset.label2id

    dataloaders = {
        "train": DataLoader(train_dataset, opt.batch_size, shuffle=True), 
        "val": DataLoader(val_dataset, opt.batch_size)
    }

    model = ViT(
        image_size=224,
        patch_size=32,
        num_classes=2,
        dim = 1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    )



    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    criterion = nn.CrossEntropyLoss()

    model, hist, f_maxtrix = train_model(model, dataloaders, criterion, optimizer, opt)

    print(hist)
    print(f_maxtrix)


