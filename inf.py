from pathlib import Path
import shutil
import torch
from src.grabcut import GrabCutProcessor
import cv2
import matplotlib.pyplot as plt
import numpy as np


class InferenceRunner:

    def __init__(self, model, class_names, device=None):
        self.model = model
        self.class_names = class_names
        self.device = device

    def _prepare_output_directory(self, output_dir):
        output_dir = Path(output_dir)

        if output_dir.exists() and output_dir.is_dir():
            for item in output_dir.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir

    def run_inference(self, dataset, output_dir):
        """
        Runs inference on a dataset, applies GrabCut refinement, and saves results.

        Args:
            dataset (Dataset): Dataset containing test images.
            output_dir (str): Directory to save inference results.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running inference on {len(dataset)} images...")

        for idx in range(len(dataset)):
            image, _, _ = dataset[idx]
            image_tensor = image.unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                predictions = self.model(image_tensor)

            # Extract masks and bounding boxes
            r = {
                "masks": predictions[0]["masks"].squeeze(1).cpu().numpy(),
                "rois": predictions[0]["boxes"].cpu().numpy(),
                "class_ids": predictions[0]["labels"].cpu().tolist(),
            }
            masks, rects = GrabCutProcessor.separateEntities(r)

            # Visualize and refine masks with GrabCut
            self.visualize_and_apply_grabcut(
                image.permute(1, 2, 0).numpy(), masks, rects, output_dir, idx
            )

    def visualize_and_apply_grabcut(self, image, masks, rects, output_dir, idx):
        """
        Visualizes masks and applies GrabCut refinement.

        Args:
            image (numpy.ndarray): Input image.
            masks (list): List of binary masks.
            rects (list): List of bounding boxes.
            output_dir (Path): Directory to save the results.
            idx (int): Index of the current image.
        """

        if len(masks) == 0:
            print(f"No masks found for image {idx + 1}. Skipping visualization.")
            return

        # Ensure image is uint8 and has 3 channels
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if len(image.shape) == 2:  # Convert grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] != 3:  # Validate channel count
            raise ValueError(
                f"Expected a 3-channel image, but got {image.shape[2]} channels."
            )

        fig, axs = plt.subplots(len(masks), 2, figsize=(10, len(masks) * 5))

        for i, (mask, rect) in enumerate(zip(masks, rects)):
            refined = GrabCutProcessor.applyGrabCut(image, mask, rect, iters=10)
            definite_fg = refined["Definite Foreground"]

            # Ensure definite_fg is 2D
        if len(definite_fg.shape) == 1:
            definite_fg = definite_fg.reshape(image.shape[:2])

            segmented = cv2.bitwise_and(image, image, mask=definite_fg)

            axs[i, 0].imshow(definite_fg, cmap="gray")
            axs[i, 0].axis("off")
            axs[i, 0].set_title(f"Refined Mask {i + 1}")

            axs[i, 1].imshow(segmented)
            axs[i, 1].axis("off")
            axs[i, 1].set_title(f"Segmented Image {i + 1}")

        # Save combined results
        combined_output_path = output_dir / f"combined_{idx + 1}.png"
        plt.savefig(combined_output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved combined visualization to {combined_output_path}")
