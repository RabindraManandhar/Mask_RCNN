# Mask_RCNN
This project implements Mask R-CNN using Python 3 and PyTorch. The model generates instance-specific segmentation masks and bounding boxes for objects in images, leveraging a Feature Pyramid Network (FPN) with a ResNet50 backbone.

## Features
- **Pre-trained Models**: Uses pre-trained Mask R-CNN with ResNet-50 backbone.
- **COCO-style Dataset Support**: Load and preprocess datasets in COCO format.
- **Training and Validation**: Easy-to-use traininig utilities with validation support.
- **Inference Pipeline**: Run inference on datasets or individual images.
- **Visualizations**: Save and view results with overlaid masks and class labels.

## Project Structure
```plaintext
Mask_RCNN/
├── data/                       # Dataset directories (not included in the package)
│   ├── train/
│   ├── test/
│   ├── valid/
├── images/                     # Directory for input images
│   └── *.JPG
├── models/                     # Directory for saving trained models
│   └── *.pth
├── src/                        # Source code for the package
│   ├── __init__.py             # Initializes the package
│   ├── datasets.py             # Dataset utilities
│   ├── inference.py            # Inference and visualization utilities
│   ├── trainer.py              # Training utilities
│   ├── utils.py                # Visualization and general utility functions
├── tests/                      # Unit tests
│   ├── __init__.py
├── visualizations_new_images/  # Directory for saving test images outputs
│   └── *.png
├── visualizations_test_images/  # Directory for saving new images outputs
│   └── *.png
├── .gitignore                  # Git ignored files
├── main.py                     # Entry point for the project
├── LICENSE                     # License file
├── pyproject.toml              # Metadata for Python packaging
├── README.md                   # Detailed project description
├── requirements.txt            # Required Python dependencies
```

## Installation

#### 1. Clone the repository
```bash
git clone https://github.com/RabindraManandhar/Mask_RCNN.git
cd Mask_RCNN
```

## Requirements

#### 1. Dataset Requirements
```plaintext
data/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── _annotations.coco.json
├── test/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── _annotations.coco.json
├── valid/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── _annotations.coco.json
```

#### 2. Dependencies Requirements
```plaintext
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
opencv-python>=4.5.0
pycocotools>=2.0.2
numpy>=1.19.0
```

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Usage
#### 1. Training the Model
```bash
python main.py
```
This script:
- Loads the dataset from the data/ directory.
- Trains the Mask R-CNN model for a specified number of epochs.
- Saves the trained model in the models/ directory.

#### 2. Running Inference
**On Test Dataset:**
Use the InferenceRunner to run inference on the test dataset:
```bash
inference_runner = InferenceRunner(
    model = trained_model,
    class_names=["Background", "Fruit", "Leaf", "Flower"],
    device="cuda"
)
inference_runner.run_inference(
    dataset=test_dataset, output_dir="visualizations_test_images/"
)
```
**On New Images:**
Run inference on all .JPG images in a directory:
```bash
inference_runner.run_inference_on_images(
    image_dir="images/", output_dir="visualizations_new_images/"
)
```

## Limitations
- The model requires a COCO-style dataset format for training and inference.
- Training times may vary based on the size of the dataset and hardware used.










