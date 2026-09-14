 
import json
import random
 
import numpy as np
import pydicom
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
 
IMG_SIZE = 518
 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
 
class InbreastMassDataset(Dataset):
    def __init__(self, split_json_path, split_name, img_size=IMG_SIZE, augment=None):
        with open(split_json_path) as f:
            records = json.load(f)
        self.records = [r for r in records if r["split"] == split_name]
        self.img_size = img_size
        self.augment = augment if augment is not None else (split_name == "train")
 
        if len(self.records) == 0:
            raise ValueError(f"Nema zapisa za split='{split_name}' u {split_json_path}")
 
    def __len__(self):
        return len(self.records)
 
    def _load_image(self, dicom_path):
        ds = pydicom.dcmread(dicom_path)
        image = ds.pixel_array.astype(np.float32)
 
        photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
        if photometric == "MONOCHROME1":
            image = image.max() - image
 
        img_min, img_max = image.min(), image.max()
        image = (image - img_min) / (img_max - img_min + 1e-6)
        return image
 
    def _load_mask(self, mask_path):
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)
        return mask
 
    def _augment(self, image_t, mask_t):
        
        if random.random() < 0.5:
            image_t = TF.hflip(image_t)
            mask_t = TF.hflip(mask_t)
 
        angle = random.uniform(-15, 15)
        image_t = TF.rotate(
            image_t, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0.0
        )
        mask_t = TF.rotate(
            mask_t, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0.0
        )
 
        gain = random.uniform(0.85, 1.15)
        bias = random.uniform(-0.1, 0.1)
        image_t = (image_t * gain + bias).clamp(0.0, 1.0)
 
        return image_t, mask_t
 
    def __getitem__(self, idx):
        rec = self.records[idx]
 
        image = self._load_image(rec["dicom_path"])
        mask = self._load_mask(rec["mask_path"])
 
        image_t = torch.from_numpy(image).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
 
        image_t = TF.resize(
            image_t, [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        mask_t = TF.resize(
            mask_t, [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.NEAREST,
        )
 
        if self.augment:
            image_t, mask_t = self._augment(image_t, mask_t)
 
        image_t = image_t.repeat(3, 1, 1)
        image_t = TF.normalize(image_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
 
        return {
            "image": image_t,
            "mask": mask_t,
            "id": rec["id"],
        }
