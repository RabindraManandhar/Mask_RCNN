import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add the 'src' directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "src"))


# Import specific classes and functions from the modules inside src/ to expose them at the package level.
from src.datasets import COCODataset
from src.trainer import MaskRCNNTrainer
from src.inference import InferenceRunner
from src.utils import get_transforms


def main():
    """
    Entry point for training, validating, and running inference with Mask R-CNN.
    """
    # Paths to data and output directories
    base_dir = Path(__file__).resolve().parent  # Root directory
    dataset_dir = base_dir / "data"
    images_dir = base_dir / "images"
    models_dir = base_dir / "models"
    inference_results_dataset_dir = base_dir / "visualizations_test_images"
    inference_results_images_dir = base_dir / "visualizations_new_images"

    # Define paths for train, validation, and test datasets
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "valid"
    test_dir = dataset_dir / "test"

    # Define paths for COCO annotation files
    train_annotations = train_dir / "_annotations.coco.json"
    val_annotations = val_dir / "_annotations.coco.json"
    test_annotations = test_dir / "_annotations.coco.json"

    # Configuration
    num_classes = 4  # including background
    num_epochs = 2
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Ensure necessary directories exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(inference_results_dataset_dir, exist_ok=True)
    os.makedirs(inference_results_images_dir, exist_ok=True)

    # Prepare datasets and data loaders
    # Training Dataset and DataLoader
    try:
        train_dataset = COCODataset(
            train_dir, train_annotations, transforms=get_transforms(augment=False)
        )
    except Exception as e:
        print(f"Training dataset loading error: {e}")
        exit()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    # Validation Dataset and DataLoader
    try:
        valid_dataset = COCODataset(
            val_dir, val_annotations, transforms=get_transforms(augment=False)
        )
    except Exception as e:
        print(f"Validation dataset loading error: {e}")
        exit()

    val_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    # Test Dataset
    try:
        test_dataset = COCODataset(
            test_dir, test_annotations, transforms=get_transforms(augment=False)
        )
    except Exception as e:
        print(f"Testing dataset loading error: {e}")
        exit()

    # Initialize the model trainer
    trainer = MaskRCNNTrainer(num_classes=num_classes, device=device)
    optimizer = torch.optim.SGD(
        params=trainer.model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
    )
    trainer.compile(optimizer)

    # Training the model
    print("Training Started...")
    """
    trainer.train(
        data_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        validate_every=1,
    )
    """

    # Save the model
    # print("Saving the model ...")
    model_path = models_dir / "mask_rcnn_model.pth"
    # trainer.save_model(model_name=model_path, model_dir="models")

    # Load the saved model
    print("Loading the trained model...")
    trainer.load_model(model_name=model_path.name, model_dir="models")

    # Run inference on the test dataset
    print("Running inference on the test dataset...")
    class_names = ["Background", "Fruit", "Leaf", "Flower"]
    inference_runner = InferenceRunner(
        trainer.model,
        class_names,
        device,
    )
    # Run inference on test dataset
    inference_runner.run_inference(
        dataset=test_dataset, output_dir=inference_results_dataset_dir
    )

    # Run inference on new .jpg images
    inference_runner.run_inference_on_images(
        image_dir=images_dir, output_dir=inference_results_images_dir
    )

    print("All tasks completed!")


if __name__ == "__main__":
    main()
