from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets import COCODataset
from trainer import MaskRCNNTrainer
from grabcut import GrabCutProcessor
from utils import get_transforms, visualize_and_save_predictions


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
    device = "cpu"

    # Dataset and DataLoader
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

    # Model Trainer
    trainer = MaskRCNNTrainer(num_classes=num_classes, device=device)
    optimizer = torch.optim.SGD(
        params=trainer.model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
    )
    trainer.compile(optimizer)

    # Train the model
    print("Training the model...")
    trainer.train(data_loader=train_loader, num_epochs=num_epochs)

    # Save the model
    trainer.save_model(model_name="mask_rcnn_model.pth")

    # Load the saved model
    trainer.load_model(model_name="mask_rcnn_model.pth")

    # Test predictions
    print("Testing the model...")
    test_dataset = COCODataset(test_dir, test_annotations, transforms=get_transforms())
    test_image, _, _ = test_dataset[0]
    test_image_tensor = test_image.unsqueeze(0).to(device)

    # Define class names
    class_names = ["Background", "Fruit", "Leaf", "Flower"]

    # Run inference on all test images
    for idx in range(len(test_dataset)):
        test_image, _, _ = test_dataset[idx]
        test_image_tensor = test_image.unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            predictions = trainer.model(test_image_tensor)

        # Extract predictions
        predicted_masks = (predictions[0]["masks"] > 0.5).squeeze(1).cpu().numpy()
        predicted_labels = predictions[0]["labels"].cpu().tolist()

        # Define output path for saving visualization
        output_path = f"visualizations/test_image_{idx + 1}.png"

        # Visualize and save predictions
        visualize_and_save_predictions(
            test_image.permute(
                1, 2, 0
            ).numpy(),  # Convert tensor to numpy array (H, W, 3)
            predicted_masks,  # Predicted masks (N, H, W)
            predicted_labels,  # Predicted class IDs
            class_names,  # List of class names
            output_path,  # Path to save visualization
        )


if __name__ == "__main__":
    main()
