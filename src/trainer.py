import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pathlib import Path
import shutil


class MaskRCNNTrainer:
    """
    A class for training and validating a Mask R-CNN model for instance segmentation.
    """

    def __init__(self, num_classes, device):

        self.model = self._initialize_model(num_classes)
        self.device = device
        self.optimizer = None

    def _initialize_model(self, num_classes):

        # Load pre-trained Mask R-CNN model with COCO weights
        weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        model = maskrcnn_resnet50_fpn(weights=weights)

        # Replace the box predictor for classification
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Replace the mask predictor for instance segmentation
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

        return model

    def compile(self, optimizer):
        # Attach optimizer to the model
        self.optimizer = optimizer

    def train(self, data_loader, val_loader=None, num_epochs=None, validate_every=1):

        self.model.to(self.device)

        for epoch in range(num_epochs):
            # Ensure model is in training mode
            self.model.train()
            total_loss = 0

            print(f"Training started for Epoch {epoch + 1}/{num_epochs} ")

            for batch_idx, (images, targets, _) in enumerate(data_loader):
                # Move images and targets to the specified device
                images = [image.to(self.device) for image in images]
                targets = [
                    {k: v.to(self.device) for k, v in target.items()}
                    for target in targets
                ]

                try:
                    # Forward pass to compute loss
                    loss_dict = self.model(images, targets)
                    print("Loss Dict:", loss_dict)

                    # Sum up all losses
                    losses = sum(loss for loss in loss_dict.values())
                    print(
                        f"Epoch {epoch + 1}, Batch {batch_idx + 1}: Loss = {losses.item():.4f}"
                    )
                    # Backward pass to optimize weights
                    self.optimizer.zero_grad()
                    losses.backward()
                    self.optimizer.step()

                    total_loss += losses.item()

                except RuntimeError as e:
                    print(
                        f"Error during training at Epoch {epoch + 1}, Batch {batch_idx + 1}: {e}"
                    )
                    # print("Loss Dict:", loss_dict)
                    exit()

            # Calculate average loss
            avg_loss = total_loss / len(data_loader)
            print(f"Epoch {epoch + 1} / {num_epochs} - Average Loss: {avg_loss:.4f}")

            # Run validation if required
            if val_loader and (epoch + 1) % validate_every == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - Running validation")
                self.validate(val_loader)

    def validate(self, val_loader):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0

        with torch.no_grad():  # No gradient calculation during validation
            for images, targets, _ in val_loader:
                images = [image.to(self.device) for image in images]
                targets = [
                    {k: v.to(self.device) for k, v in target.items()}
                    for target in targets
                ]

                # Temporarily switch to training mode for loss computation
                self.model.train()
                try:
                    loss_dict = self.model(images, targets)  # Compute loss
                    batch_loss = sum(loss for loss in loss_dict.values())
                    total_loss += batch_loss.item()
                except Exception as e:
                    print(f"Error during validation: {e}")
                finally:
                    self.model.eval()  # Restore evaluation mode

        avg_loss = total_loss / len(val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")

    def save_model(self, model_name, model_dir=None):
        # Create model directory
        model_dir = Path(model_dir)

        # Empty the model_dir directory if exists or create model_dir if not exist
        if model_dir.exists() and model_dir.is_dir():
            for item in model_dir.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        else:
            model_dir.mkdir(exist_ok=True, parents=True)

        # Define the full save path
        model_save_path = model_dir / model_name

        # Save the model's state dictionary
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")

    def load_model(self, model_name, model_dir=None):
        # Define the full path to the model file
        model_path = Path(model_dir) / model_name

        # Check if the file exists
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        # Load the model's state dictionary
        self.model.load_state_dict(torch.load(model_path))
        # Set model to evaluation mode
        self.model.eval()
        print(f"Model loaded from {model_path}")

        # Return the model with loaded weights
        return self.model
