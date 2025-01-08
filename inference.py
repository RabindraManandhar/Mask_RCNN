from pathlib import Path
import shutil
import torch
from utils import visualize_and_save_predictions


class InferenceRunner:
    def __init__(self, model, class_names, device=None):
        self.model = model
        self.class_names = class_names
        self.device = device

    def run_inference(self, dataset, output_dir):
        output_dir = Path(output_dir)

        if output_dir.exists() and output_dir.is_dir():
            for item in output_dir():
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running inference on {len(dataset)} images...")

        for idx in range(len(dataset)):
            test_image, _, _ = dataset[idx]
            test_image_tensor = test_image.unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                predictions = self.model(test_image_tensor)

            # Extract predictions
            predicted_masks = (predictions[0]["masks"] > 0.5).squeeze(1).cpu().numpy()
            predicted_labels = predictions[0]["labels"].cpu().tolist()

            # Define output path
            output_path = output_dir / f"test_image_{idx + 1}.png"

            # Visualize and save predictions
            visualize_and_save_predictions(
                test_image.permute(
                    1, 2, 0
                ).numpy(),  # Convert tensor to numpy array (H, W, 3)
                predicted_masks,  # Predicted masks (N, H, W)
                predicted_labels,  # Predicted class IDs
                self.class_names,
                str(output_path),
            )
        print(f"Visualizations saved in: {output_dir}")
