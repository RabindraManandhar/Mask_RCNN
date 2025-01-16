import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import cv2
import numpy as np


class COCODataset(Dataset):
    def __init__(self, images_dir, annotation_file, transforms=None):
        self.coco = COCO(annotation_file)  # Load COCO annotations
        self.images_dir = images_dir  # Path to the image directory
        self.image_ids = list(self.coco.imgs.keys())  # List of image IDs
        self.transforms = transforms  # Transformations (if any)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Get the COCO image ID for the given index
        image_id = self.image_ids[idx]

        # Load annotation for the image
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        # Get the full path to the image file
        image_path = os.path.join(
            self.images_dir, self.coco.imgs[image_id]["file_name"]
        )

        # Ensure the image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert the image from BGR (OpenCV format) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert image to PIL format for transformations
        image = Image.fromarray(image)

        # Extract bounding boxes, labels, and masks from annotations
        boxes, labels, masks = [], [], []

        for ann in annotations:
            # Extract bounding box: [xmin, ymin, xmax, ymax]
            xmin, ymin, width, height = ann["bbox"]
            boxes.append([xmin, ymin, xmin + width, ymin + height])

            # Extract category label
            labels.append(ann["category_id"])

            # Generate segmentation mask (if available)
            if "segmentation" in ann and len(ann["segmentation"]) > 0:
                mask = self.coco.annToMask(ann)
                masks.append(mask)

        # Stack masks into a single array, or use a placeholder if no masks exist
        masks = np.stack(masks, axis=0) if masks else np.zeros((1, *image.shape[:2]))

        # Convert bounding boxes, labels, and masks to PyTorch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Create a target dictionary with annotations
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
            if not isinstance(image, Image.Image):  # Ensure input is a PIL image
                raise TypeError(f"Expected a PIL.Image, but got {type(image)}")
            image = self.transforms(image)

        return image, targets, annotations
