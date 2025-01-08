import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from pathlib import Path


class MaskRCNNTrainer:
    def __init__(self, num_classes, device):
        self.model = self._initialize_model(num_classes)
        self.device = device
        self.optimizer = None

    def _initialize_model(self, num_classes):
        # Use explicit weights for pre-trained model
        weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        model = maskrcnn_resnet50_fpn(weights=weights)

        # Replace the box classifier (classification head)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Replace the mask predictor (mask head)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

        return model

    def compile(self, optimizer):
        # Attach optimizer to the model
        self.optimizer = optimizer

    def train(self, data_loader, num_epochs):
        self.model.to(self.device)
        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0
            for images, targets, _ in data_loader:
                images = list(image.to(self.device) for image in images)
                targets = [
                    {k: v.to(self.device) for k, v in target.items()}
                    for target in targets
                ]

                try:
                    # Forward pass
                    loss_dict = self.model(images, targets)
                    print("Loss Dict:", loss_dict)

                    # Ensure all losses are contiguous
                    for loss_name, loss_value in loss_dict.items():
                        if not loss_value.is_contiguous():
                            print(
                                f"{loss_name} is not contiguous. Making it contiguous."
                            )
                            loss_dict[loss_name] = loss_value.contiguous()

                    # Sum the losses
                    losses = sum(loss.reshape(1) for loss in loss_dict.values())
                    print(f"Total Loss for Batch: {losses.item()}")

                    # Backward pass
                    self.optimizer.zero_grad()
                    losses.backward()
                    self.optimizer.step()

                    total_loss += losses.item()

                except RuntimeError as e:
                    print(f"Error during training: {e}")
                    print("Loss Dict:", loss_dict)
                    exit()

            print(f"Epoch {epoch + 1} / {num_epochs}, Loss: {total_loss:.4f}")

    def save_model(self, model_name, model_dir="models"):
        # Create model directory
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True, parents=True)

        # Define the full save path
        model_save_path = model_path / model_name

        # Save the model's state dictionary
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")

    def load_model(self, model_name, model_dir="models"):
        # Define the full path to the model file
        model_path = Path(model_dir) / model_name

        # Check if the file exists
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        # Load the model's state dictionary
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print(f"Model loaded from {model_path}")

        # Return the model with loaded weights
        return self.model
