import os
import torch
import shutil
import torchvision
from PIL import Image
import torch.nn as nn
from model.vae import VAE
import matplotlib.pyplot as plt
from model.noise import NoiseScheduler 
import torchvision.transforms.v2 as v2
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import IterableDataset, DataLoader 
import torch.distributed as dist

effects = v2.Compose([
    v2.RandomRotation(degrees=(0, 60)),
    v2.RandomHorizontalFlip(),
    v2.RandomPerspective(distortion_scale = 0.4, p = 0.6),
    v2.Resize(size = (384, 384)), # for vae
    ToTensor(),
    v2.Lambda(lambda t: t * 2 - 1) # from [0, 1] to [-1, 1]
])

class OptimizeImage(nn.Module):
    " optimizes an image and makes it trainable "
    def __init__(self):
        super(OptimizeImage, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.tensor:
        # self.opt_image = nn.Parameter(image.clone().detach())
        self.opt_image = nn.Parameter(image.clone().detach().to(image.device))
        return self.opt_image

class NamedImageFolder(ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        # convert index to class name
        class_name = self.classes[label]
        return image, class_name  
    
# class NamedImageFolder(ImageFolder):
#     def __getitem__(self, index):
#         image, label_idx = super().__getitem__(index)
#         class_name = self.classes[label_idx]
#         return image, class_name  # 여기에 label_idx를 포함하지 않으면 문제됨
    
class ImageDataset(IterableDataset):
    def __init__(self, image_dataset: NamedImageFolder, transforms=None, batch: int=4, max_samples=12, device="cpu"):
        super(ImageDataset, self).__init__()
        
        self.images = image_dataset.samples # list (img_path, label_index)
        self.labels = image_dataset.classes

        self.transforms = transforms
        self.scheduler = NoiseScheduler()
        self.make_trainable = OptimizeImage()
        self.device = device
        self.vae = VAE().to(self.device).eval() 

        self.batch_size = batch
        self.max_samples = max_samples
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    def __len__(self):
        return self.max_samples

    def __iter__(self):
        batch = []
        sample_count = 0

        for i, (image_path, label_idx) in enumerate(self.images):
            if (i % self.world_size) != self.rank:
                continue
            if sample_count >= self.max_samples:
                break

            try:
                # print(f"[RANK {self.rank}] Loading image {image_path}", flush=True)
                image = Image.open(image_path).convert("RGB")
                if self.transforms:
                    image = self.transforms(image)
                image = image.unsqueeze(0).to(self.device)  # GPU로 이동

                # print(f"[RANK {self.rank}] Before VAE encode", flush=True)
                latent_dist = self.vae.encode(image)
                # print(f"[RANK {self.rank}] After VAE.encode()", flush=True)

                latent = latent_dist.sample().to(self.device)  # GPU로 이동
                # print(f"[RANK {self.rank}] After latent.sample()", flush=True)

                noise, added_noise, timestep, sigma = self.scheduler.add_noise(latent.unsqueeze(0))
                noise = noise.to(self.device)
                added_noise = added_noise.to(self.device)
                timestep = timestep.to(self.device)
                sigma = sigma.to(self.device)

                # print(f"[RANK {self.rank}] After noise injection", flush=True)

                noise = self.make_trainable(noise).to(self.device)
                # print(f"[RANK {self.rank}] After make_trainable", flush=True)

                # label_idx가 유효한 인덱스인지 방어적으로 확인
                if label_idx >= len(self.labels):
                    # print(f"[RANK {self.rank}] Warning: label index {label_idx} out of range", flush=True)
                    continue
                
                # label_tensor = torch.tensor([label_idx], dtype=torch.long, device=self.device)
                # batch.append((latent, noise, added_noise, timestep, label_tensor))
                label = self.labels[label_idx]

                batch.append((latent, noise, added_noise, timestep, label, sigma))
                sample_count += 1

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

            except Exception as e:
                print(f"[RANK {self.rank}] (-) Error: {e}", flush=True)

# def check_dataset_dir(type: str):
#     # check if the dir of dataset is valid
#     path = os.path.join(os.getcwd(), type)
#     assert os.path.exists(path), "{type} should exist in working directory miniDiffusion or should be passed as a parameter"
    
#     return True

def check_dataset_dir(path: str):
    assert os.path.exists(path), f"Dataset path '{path}' does not exist."
    return True

def get_dataloader(dataset, device: str = "cpu") -> DataLoader:
    """ Returns an optimized DataLoader based on the computing device. """
    
    # enable pin memory for gpu
    pin_memory = device == "cuda"

    return DataLoader(
        dataset,
        batch_size = 1, # batch size one for IterableDataset
        num_workers = 0,
        pin_memory = pin_memory,
        prefetch_factor = None,
        persistent_workers = False,
    )

# def get_dataset(path: str, batch_size: int, device: str = "cpu") -> DataLoader:
#     " Get dataset ready "

#     dataset = NamedImageFolder(path)
#     dataset = ImageDataset(dataset, batch = batch_size, transforms = effects)
#     dataloader = get_dataloader(dataset, device = device)

#     return dataloader

def get_dataset(path: str, batch_size: int, device: str = "cpu") -> DataLoader:
    dataset = NamedImageFolder(path)
    dataset = ImageDataset(dataset, batch=batch_size, transforms=effects, device=device)
    dataloader = get_dataloader(dataset, device=device)
    return dataloader

def get_fashion_mnist_dataset(batch_size: int = 1, device: str = "cpu") -> DataLoader:
    """
    Get Fashion MNIST dataset ready for train.py script
    """
    #  base directory
    base_dir = os.path.join(os.getcwd(), "data", "fashion_mnist")
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):

        train_dataset = torchvision.datasets.FashionMNIST(root="./data", train = True, download = True)
        test_dataset = torchvision.datasets.FashionMNIST(root="./data", train = False, download = True)

        class_names = train_dataset.classes 

        for class_name in class_names:
            os.makedirs(os.path.join(train_dir, class_name), exist_ok = True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok = True)

        def save_images(dataset, split):
            # save images into folder
            root_dir = train_dir if split == "train" else test_dir

            for idx in range(len(dataset)):
                img, label = dataset[idx]
                class_name = class_names[label]
                img_path = os.path.join(root_dir, class_name, f"{idx}.png")
                img.save(img_path)

        # Save train and test sets
        save_images(train_dataset, "train")
        save_images(test_dataset, "test")

        shutil.rmtree(os.path.join(os.getcwd(), "data", "FashionMNIST"))

    train_dataset = get_dataset(train_dir, batch_size = batch_size, device = device)
    test_dataset = get_dataset(test_dir, batch_size = batch_size, device = device)

    return train_dataset, test_dataset

def test_fashion():
    train_dataset, _ = get_fashion_mnist_dataset()

    for batch in train_dataset: # (_, _, image, label, _)
        image = batch[0][2].squeeze(0).squeeze(0)
        label = batch[0][3]

        plt.figure()
        plt.imshow(image.squeeze(0).permute(1, 2, 0).detach().to(torch.float32).numpy()) 
        plt.title(label.squeeze(0))
        plt.show()
        break