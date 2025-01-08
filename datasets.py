import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2
import numpy as np


class COCODataset(Dataset):
    def __init__(self, images_dir, annotation_file, transforms=None):
        self.coco = COCO(annotation_file)
        self.images_dir = images_dir
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Get image id
        image_id = self.image_ids[idx]

        # Load annotation
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        # Load image
        image_path = os.path.join(
            self.images_dir, self.coco.imgs[image_id]["file_name"]
        )

        # Check if the image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract masks
        boxes = []
        labels = []
        masks = []

        for ann in annotations:
            xmin, ymin, width, height = ann["bbox"]
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(ann["category_id"])

            # Add segmentation mask
            if "segmentation" in ann and len(ann["segmentation"]) > 0:
                mask = self.coco.annToMask(ann)
                masks.append(mask)

        masks = np.stack(masks, axis=0) if masks else np.zeros((1, *image.shape[:2]))

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Create target dictionary
        targets = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([image_id]),
            "area": torch.tensor(
                [ann["area"] for ann in annotations], dtype=torch.float32
            ),
            "iscrowd": torch.tensor(
                [ann["iscrowd"] for ann in annotations], dtype=torch.int64
            ),
        }
        # print(targets)

        # Apply transforms if provided
        if self.transforms:
            image = self.transforms(image)

        return image, targets, annotations
