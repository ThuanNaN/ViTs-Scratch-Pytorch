# Implement ViT from scratch for Image Classification.

## 1. Data Structure

```
data_dir/
    train/
        class_1/
        class_2/
        ...
    val/
        class_1/
        class_2/
        ...
    test/
        class_1/
        class_2/
        ...
```

## 2. Install packages
```
pip3 install -r requirements.txt
```

## 3. Config 
### 3.1 Config data training
Setting it in config/data_config.yaml

Example:
```
train: ./data/train
val: ./data/val
test: ./data/test
n_classes: 2
```

### 3.2 Config model training
Setting it in config/train_config.yaml

Example:
```
device: mps
n_epochs: 5
batch_size: 16
learning_rate: 0.0002
weight_decay: 0.0002
PATH_SAVE: ./ckpt
model_name: google/vit-base-patch32-224-in21k
default_data_transform: True 

patch_size: 32
dim: 1024
depth: 6
heads: 8
mlp_dim: 2048
dropout: 0.1
emb_dropout: 0.1
```

### Note: device for Apple Silicon will be 'mlp', normal will be in {'cpu', 'cuda'}

## 4. Run
To train model, just run this command:
```
python train.py
```

## 5. Reference
- Repo: https://github.com/lucidrains/vit-pytorch
