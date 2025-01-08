from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets import COCODataset
from trainer import MaskRCNNTrainer
from inference import InferenceRunner
from grabcut import GrabCutProcessor
from utils import get_transforms


def main():
    # Define dataset directory as a Path object
    dataset_dir = Path("dataset")

    # Define paths for train, validation, and test directories
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "valid"
    test_dir = dataset_dir / "test"

    # Define paths for COCO annotation files
    train_annotations = train_dir / "_annotations.coco.json"
    val_annotations = val_dir / "_annotations.coco.json"
    test_annotations = test_dir / "_annotations.coco.json"
    num_classes = 4  # fruit, flower, leaves
    num_epochs = 2

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Training Dataset and DataLoader
    try:
        train_dataset = COCODataset(
            train_dir, train_annotations, transforms=get_transforms()
        )
    except Exception as e:
        print(f"Dataset loading error: {e}")
        exit()

    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )

    # Validation Dataset and DataLoader
    try:
        valid_dataset = COCODataset(
            val_dir, val_annotations, transforms=get_transforms()
        )
    except Exception as e:
        print(f"Validation dataset loading error: {e}")
        exit()

    val_loader = DataLoader(
        valid_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x))
    )

    # Model Trainer
    trainer = MaskRCNNTrainer(num_classes=num_classes, device=device)
    optimizer = torch.optim.SGD(
        params=trainer.model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
    )
    trainer.compile(optimizer)

    # Train the model
    print("Training the model...")
    trainer.train(
        data_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        validate_every=1,
    )

    # Save the model
    print("Saving the model ...")
    trainer.save_model(model_name="mask_rcnn_model.pth", model_dir="models")

    # Load the saved model
    trainer.load_model(model_name="mask_rcnn_model.pth", model_dir="models")

    # Test Dataset and DataLoader
    try:
        test_dataset = COCODataset(
            test_dir, test_annotations, transforms=get_transforms()
        )
    except Exception as e:
        print(f"Testing dataset loading error: {e}")
        exit()

    # Run inference
    class_names = ["Background", "Fruit", "Leaf", "Flower"]
    inference_runner = InferenceRunner(trainer.model, class_names, device)

    # Run inference on test dataset
    inference_runner.run_inference(dataset=test_dataset, output_dir="visualizations")

    # Run inference on new .jpg images
    image_dir = "images"
    inference_runner.run_inference_on_images(
        image_dir=image_dir, output_dir="inference_results"
    )


if __name__ == "__main__":
    main()
