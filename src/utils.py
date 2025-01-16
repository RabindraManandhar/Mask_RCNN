import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import os


def get_transforms(augment=False):
    """
    Returns a torchvision transformation pipeline for preprocessing images.

    Args:
        augment (bool): If True, applies data augmentation for training.

    Returns:
        torchvision.transforms.Compose: A transformation pipeline.
    """
    # Define the transformation steps
    transforms = []

    # Resize the image (optional, useful for training consistency)
    transforms.append(T.Resize((512, 512)))  # Resize to fixed dimensions (H, W)

    # Apply data augmentation if specified
    if augment:
        transforms.extend(
            [
                T.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
                T.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                ),  # Randomly adjust color properties
                T.RandomRotation(degrees=10),  # Small random rotation
                T.RandomApply([T.GaussianBlur(kernel_size=(5, 5))], p=0.3),  # Blur
            ]
        )

    # Convert the image to a PyTorch tensor
    transforms.append(T.ToTensor())

    # Normalize the image (use COCO mean/std or ImageNet mean/std as applicable)
    transforms.append(
        T.Normalize(
            mean=[0.485, 0.456, 0.406],  # COCO or ImageNet mean
            std=[0.229, 0.224, 0.225],  # COCO or ImageNet std
        )
    )

    return T.Compose(transforms)


def visualize_and_save_predictions(image, masks, labels, class_names, output_path):

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize the plot
    plt.figure(figsize=(8, 6))
    plt.imshow(image)

    # Overlay each mask corresponding to object classes (labels > 0) and annotate class names
    for i, mask in enumerate(masks):
        # Skip background masks (labels with ID 0)
        if labels[i] > 0:
            plt.imshow(mask, alpha=0.5, cmap="jet")  # Overlay mask with transparency

            # Annotate with the class name
            mask_coords = np.argwhere(mask > 0)  # Find non-zero mask coordinates
            if len(mask_coords) > 0:
                y, x = mask_coords[0]  # Get a sample point from the mask for annotation
                plt.text(
                    x,
                    y,
                    class_names[labels[i]],  # Get the class name for the label
                    color="white",
                    fontsize=12,
                    bbox=dict(
                        facecolor="black", alpha=0.5
                    ),  # Add a background for readability
                )

    # Add a background for readability
    plt.axis("off")

    # Save the visualization to the specified output path
    plt.savefig(output_path, bbox_inches="tight")
    # Close the plot to release memory
    plt.close()
    print(f"Visualization saved to {output_path}")
