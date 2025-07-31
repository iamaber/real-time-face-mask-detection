# Real-Time Face Mask Detection

This project implements a real-time face mask detection system using PyTorch and MobileNetV3. The model is trained using transfer learning to detect whether a person is wearing a face mask or not. 

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.7+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.78+-yellow.svg)
![Gradio](https://img.shields.io/badge/Gradio-v3.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Features

- **Real-time Detection**: Process video streams and webcam feeds in real-time
- **High Accuracy**: Achieves ~96% test accuracy with robust performance
- **Web Interface**: User-friendly Gradio web application for easy interaction
- **Multiple Input Types**: Support for images, videos, and live camera feeds
- **Efficient Model**: Uses MobileNetV3 for fast inference on various devices
- **Transfer Learning**: Leverages pre-trained ImageNet weights for better performance

## 📊 Datasets

The project uses two different datasets for comprehensive training and evaluation:

1. **Primary Dataset**: [Face Mask 12K Images Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)
   - 12,000 images
   - Balanced dataset with masked and unmasked faces
   - Split into Train/Validation/Test sets
   - Classes:
     - ✅ **With Mask**: Properly worn face masks
     - ❌ **Without Mask**: No face mask detected

2. **Secondary Dataset**: [Face Mask Detection Dataset](https://www.kaggle.com/datasets/wobotintelligence/face-mask-detection-dataset)
   - Used for additional testing and validation
   - Real-world scenarios and diverse conditions
   - Different lighting conditions and camera angles

## 🚀 Quick Start

### Prerequisites
- Python 3.12+ (see [.python-version](.python-version))
- CUDA-compatible GPU (optional, for faster training)

### Installation

#### Using uv (Recommended)

```bash
# Install uv if you haven't already
pip install uv

# Clone the repository
git clone https://github.com/iamaber/real-time-face-mask-detection.git
cd real-time-face-mask-detection

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
uv pip install -r requirements.txt
```

#### Using pip

```bash
# Clone the repository
git clone https://github.com/iamaber/real-time-face-mask-detection.git
cd real-time-face-mask-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 🎮 Usage

#### Web Interface (Easiest)
```bash
python app/gradio_app.py
```
Then open your browser and navigate to the displayed URL (typically http://localhost:7860).

#### Command Line Predictions

**For Images:**
```bash
python scripts/predict_image.py --image_path /path/to/image.jpg
```

**For Videos:**
```bash
python scripts/predict_video.py --video_path /path/to/video.mp4
```

## 📈 Model Performance

### Training Metrics
- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~97%
- **Test Accuracy**: ~96%

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
- **Precision**: 0.96
- **Recall**: 0.95
- **F1-Score**: 0.955
- **ROC-AUC**: 0.98

### Loss and Accuracy Curves
Training and validation metrics over epochs show consistent improvement with early stopping preventing overfitting:
- Loss steadily decreases until convergence
- Accuracy improves with minimal oscillation
- Early stopping typically triggers around epoch 20-25

## 🏗️ Architecture Overview

The system uses **MobileNetV3-Large** as the base model with transfer learning:

- **Pre-trained Backbone**: MobileNetV3-Large (frozen feature extractor)
- **Custom Classifier**: Fully connected layers for mask detection
- **Loss Function**: Focal Loss for handling class imbalance
- **Optimization**: Early stopping and learning rate scheduling

### Model Architecture
```python
# Custom classifier head
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

## 📁 Project Structure

```
real-time-face-mask-detection/
├── 📱 app/                          # Web application
│   ├── __init__.py
│   ├── gradio_app.py               # Gradio web interface
│   └── main.py                     # FastAPI main application
├── 🤖 models/                      # Trained models
│   └── best_mobilenet_mask_detector.pt  # Best trained model
├── 📓 notebook/                    # Jupyter notebooks
│   └── Mask Detection.ipynb       # Training and analysis notebook
├── 🔧 scripts/                     # Utility scripts
│   ├── __init__.py
│   ├── predict_image.py            # Image prediction script
│   ├── predict_video.py            # Video prediction script
│   └── utils.py                    # Utility functions
├── 🎯 train.py                     # Model training script
├── 📋 requirements.txt             # Project dependencies
├── ⚙️ pyproject.toml              # Project configuration
├── 🔒 uv.lock                     # Dependency lock file
└── 📖 README.md                    # This file
```

## 🎓 Training Your Own Model

### 1. Prepare Your Data
Update the data paths in the `Config` class within [`train.py`](train.py):

```python
class Config:
    data_dir = "/path/to/your/face-mask-dataset"
    test_data_dir = "/path/to/your/test-dataset"
    # ... other configurations
```

### 2. Start Training
```bash
python train.py
```

The training process includes:
- ✨ **Data Augmentation**: Rotation, flipping, color jittering for better generalization
- 🛑 **Early Stopping**: Prevents overfitting (patience: 7 epochs)
- 📉 **Learning Rate Scheduling**: Adaptive learning rate adjustment
- 💾 **Model Checkpointing**: Automatic saving of best performing model

### 3. Monitor Training
The script automatically:
- Evaluates on validation set after each epoch
- Saves the best model based on validation accuracy to [`models/best_mobilenet_mask_detector.pt`](models/best_mobilenet_mask_detector.pt)
- Provides detailed training logs and metrics

## ⚙️ Configuration

Key hyperparameters (modifiable in [`train.py`](train.py)):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 32 | Training batch size |
| Learning Rate | 1e-4 | Initial learning rate |
| Epochs | 30 | Maximum training epochs |
| Dropout Rate | 0.5 | Dropout probability |
| Weight Decay | 1e-4 | L2 regularization |
| Early Stopping Patience | 7 | Epochs to wait before stopping |
| Focal Loss Alpha | 1.0 | Class weighting parameter |
| Focal Loss Gamma | 2.0 | Focusing parameter |

## 🛠️ Development

### Explore the Training Process
Check out the detailed analysis in [`notebook/Mask Detection.ipynb`](notebook/Mask Detection.ipynb) for:
- Data exploration and visualization
- Model architecture details
- Training metrics and plots
- Performance analysis

### Dependencies
Core dependencies (see [`requirements.txt`](requirements.txt) for complete list):
- **PyTorch** 2.7+: Deep learning framework
- **torchvision**: Computer vision utilities
- **OpenCV**: Image and video processing
- **Gradio**: Web interface framework
- **FastAPI**: API framework
- **NumPy**: Numerical computing
- **Pillow**: Image processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MobileNetV3 architecture from Google Research
- Face Mask datasets from Kaggle community
- PyTorch team for the excellent deep learning framework

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/iamaber/real-time-face-mask-detection/issues) page
2. Create a new issue with detailed description
3. Contact the maintainers

---

⭐ **Star this repository if it helped you!**
