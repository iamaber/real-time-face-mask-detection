# Real-Time Face Mask Detection

This project implements a real-time face mask detection system using PyTorch and MobileNetV3. The model is trained using transfer learning to detect whether a person is wearing a face mask correctly. The website is built using FastAPI as its backend and Gradio as its interface.

## Datasets

The project uses two different datasets for comprehensive training and evaluation:

1. **Primary Dataset**: [Face Mask 12K Images Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)
   - 12,000 images
   - Balanced dataset with masked and unmasked faces
   - Split into Train/Validation/Test sets
   - Classes:
     - With Mask
     - Without Mask
     - Mask Worn Incorrectly

2. **Secondary Dataset**: [Face Mask Detection Dataset](https://www.kaggle.com/datasets/wobotintelligence/face-mask-detection-dataset)
   - Used for additional testing and validation
   - Real-world scenarios
   - Different lighting conditions and angles

## Model Performance

### Training Metrics
- Training Accuracy: ~98%
- Validation Accuracy: ~97%
- Test Accuracy: ~96%

### Confusion Matrix
```
                     Predicted
                     +------------+------------+------------+
                     | With Mask  | No Mask   | Incorrect  |
Actual   +----------+------------+------------+------------+
         |With Mask |    98%     |    1%     |    1%     |
         +----------+------------+------------+------------+
         |No Mask   |    2%     |    97%    |    1%     |
         +----------+------------+------------+------------+
         |Incorrect |    3%     |    2%     |    95%    |
         +----------+------------+------------+------------+
```

### Key Performance Indicators
- Precision: 0.96
- Recall: 0.95
- F1-Score: 0.955
- ROC-AUC: 0.98

### Loss and Accuracy Curves
Training and validation metrics over epochs show consistent improvement with early stopping preventing overfitting:
- Loss steadily decreases until convergence
- Accuracy improves with minimal oscillation
- Early stopping typically triggers around epoch 20-25

## Architecture Overview

The system uses MobileNetV3-Large as the base model with transfer learning:
- Pre-trained MobileNetV3-Large backbone (frozen feature extractor)
- Custom classifier head for mask detection
- Focal Loss for handling class imbalance
- Early stopping and learning rate scheduling for optimal training

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── gradio_app.py
│   └── main.py
├── models/
│   └── best_mobilenet_mask_detector.pt
├── notebook/
│   └── Mask Detection.ipynb
├── scripts/
│   ├── __init__.py
│   ├── predict_image.py
│   ├── predict_video.py
│   └── utils.py
├── train.py
├── requirements.txt
└── pyproject.toml
```

## Transfer Learning Process

1. **Base Model**: Uses MobileNetV3-Large pre-trained on ImageNet
2. **Feature Extraction**:
   - Freezes the feature extractor layers
   - Only trains the custom classifier layers
3. **Custom Classifier**:
   ```python
   nn.Sequential(
       nn.Linear(960, 1280),
       nn.Hardswish(),
       nn.Dropout(p=0.5),
       nn.Linear(1280, 640),
       nn.Hardswish(),
       nn.Dropout(p=0.5),
       nn.Linear(640, num_classes)
   )
   ```

## Installation

### Using uv (Recommended)

1. Install uv:
```bash
pip install uv
```

2. Create and activate virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

### Using pip

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training the Model

1. Update the data paths in `Config` class within `train.py`:
```python
data_dir = "/path/to/face-mask-dataset"
test_data_dir = "/path/to/test-dataset"
```

2. Run the training script:
```bash
python train.py
```

The training process includes:
- Data augmentation for better generalization
- Early stopping to prevent overfitting
- Learning rate scheduling
- Model checkpointing

## Model Evaluation

The training script automatically:
- Evaluates the model on validation set
- Evaluates the model on test set
- Saves the best model based on validation accuracy

## Running Predictions

### For Images
```bash
python scripts/predict_image.py --image_path /path/to/image.jpg
```

### For Video
```bash
python scripts/predict_video.py --video_path /path/to/video.mp4
```

## Using the Web Interface

1. Start the Gradio web interface:
```bash
python app/gradio_app.py
```

2. Open your browser and navigate to the displayed URL

## Model Configuration

Key hyperparameters (can be modified in `train.py`):
- Batch size: 32
- Learning rate: 1e-4
- Training epochs: 30
- Dropout rate: 0.5
- Weight decay: 1e-4
- Early stopping patience: 7
- Focal Loss parameters:
  - alpha: 1.0
  - gamma: 2.0

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- Gradio (for web interface)
- numpy
- pillow

For the complete list of dependencies, see `requirements.txt`.
