import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os


def get_transforms():
    def transform(image):
        image = F.to_tensor(image)
        return image

    return transform


def visualize_and_save_predictions(image, masks, labels, class_names, output_path):

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(image)

    # Overlay each mask corresponding to object classes (labels > 0)
    for i, mask in enumerate(masks):
        if labels[i] > 0:  # Exclude background
            plt.imshow(mask, alpha=0.5, cmap="jet")

            # Annotate with class name
            mask_coords = np.argwhere(mask > 0)
            if len(mask_coords) > 0:
                y, x = mask_coords[0]  # Get a sample point from the mask
                plt.text(
                    x,
                    y,
                    class_names[labels[i]],
                    color="white",
                    fontsize=12,
                    bbox=dict(facecolor="black", alpha=0.5),
                )

    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {output_path}")
