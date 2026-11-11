# Real-Time Face Mask Detection

This project implements a real-time face mask detection system using PyTorch and MobileNetV3. The model is trained using transfer learning to detect whether a person is wearing a face mask or not. 

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.7+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.116+-yellow.svg)
![Gradio](https://img.shields.io/badge/Gradio-v5.38+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Live demo:** [huggingface.co/spaces/iamaber/face-mask-detector](https://huggingface.co/spaces/iamaber/face-mask-detector)

## 🎯 Features

- **Real-time Detection**: Process video streams and webcam feeds in real-time
- **High Accuracy**: 98.99% test accuracy on the 12K dataset (see [Model Performance](#-model-performance))
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
   - Used for additional testing only (evaluated, never trained on)
   - Real-world scenarios and diverse conditions
   - Different lighting conditions and camera angles
   - 7,553 images; accuracy drops to 87.97% here, showing the domain gap

## 🚀 Quick Start

### Prerequisites
- Python 3.12+ (see [.python-version](.python-version))
- CUDA-compatible GPU (optional, for faster training)

### Installation


[`pyproject.toml`](pyproject.toml) plus [`uv.lock`](uv.lock) are the single source of
truth for dependencies. Do not install from a hand-written requirements file: the lock
file is what pins exact versions. The project is not a pip-installable package (it
declares no build backend), so `uv sync` is the only supported install route.

```bash
# Install uv if you haven't already
pip install uv

# Clone the repository
git clone https://github.com/iamaber/real-time-face-mask-detection.git
cd real-time-face-mask-detection

# Create the virtual environment and install the locked dependencies
uv sync
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

### 🎮 Usage

#### Web Interface (Easiest)
```bash
python app/gradio_app.py
```
Then open your browser and navigate to the displayed URL (typically http://localhost:7860).
The script starts the FastAPI backend on port 8000, waits for its `/health` endpoint to
respond, then launches the Gradio UI. Override the addresses with
`MASK_DETECTOR_API_HOST`, `MASK_DETECTOR_API_PORT` and `MASK_DETECTOR_GRADIO_HOST`.

#### Command Line Predictions

**For Images:**
```bash
python scripts/predict_image.py --image_path /path/to/image.jpg
```

**For Videos:**
```bash
python scripts/predict_video.py --video_path /path/to/video.mp4
```
Pass a camera index (for example `--video_path 0`) to read from a webcam. Press `q` in the
preview window to stop.

Both scripts accept `--model_path` to point at a different checkpoint. By default they load
`models/best_mobilenet_mask_detector.pt`, or the path in `MASK_DETECTOR_MODEL_PATH`.

> **Model weights:** the checkpoint is stored with Git LFS. Run `git lfs install` once and
> `git lfs pull` after cloning, otherwise the file is a small pointer stub rather than
> weights. Without it, inference fails on load and the API reports `model_loaded: false`.

## 📈 Model Performance

These are the numbers printed by the committed run in
[`notebook/Mask Detection.ipynb`](notebook/Mask%20Detection.ipynb) (cells 13, 17 and 18).
They were not re-verified in this repository; treat them as the notebook's reported output.

### Reported metrics

| Split | Accuracy | Source |
|-------|----------|--------|
| Validation (800 images) | 99.50% | notebook cell 17 |
| Test (992 images) | 98.99% | notebook cell 17 |
| Secondary dataset (7,553 images) | 87.97% | notebook cell 18 |

Per-class precision and recall for the test split were both 0.99, with identical 0.99
scores for the validation split.

### Training run

The committed output trained for **10 of 25** epochs and then stopped: validation accuracy
peaked at epoch 3 (99.50%) and early stopping fired after 7 epochs without improvement.
The best checkpoint is therefore from epoch 3, not from the end of the run.

Confusion matrices for both splits are plotted in the notebook as heatmaps (cells 17-18)
with per-cell counts and percentages, rather than as a table here — that avoids quoting
numbers the notebook did not print.

### Loss and Accuracy Curves
Training and validation metrics over epochs are plotted in notebook cell 14:
- Training loss falls from 0.0308 (epoch 1) to 0.0083 (epoch 10)
- Validation accuracy stays between 98.75% and 99.50%
- Early stopping triggered after epoch 10, not "around epoch 20-25"

## 🏗️ Architecture Overview

The system uses **MobileNetV3-Large** as the base model with transfer learning:

- **Pre-trained Backbone**: MobileNetV3-Large (frozen feature extractor)
- **Custom Classifier**: Fully connected layers for mask detection
- **Loss Function**: Focal Loss for hard-example mining (see the note below)
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
├── 🎯 train.py                     # Model training entry point
├── ⚙️ config.py                    # Shared Config (training + inference)
├── 📋 pyproject.toml              # Project dependencies
├── 🔒 uv.lock                     # Dependency lock file
└── 📖 README.md                    # This file
```

> **On the loss function:** the dataset is balanced rather than imbalanced — the notebook
> reports exactly 400/400 validation samples (cell 17) and 483/509 test samples — and
> `focal_alpha = 1.0` makes the alpha term uniform, so alpha does nothing here. The
> useful part is the `(1 - pt) ** gamma` factor from `gamma = 2.0`, which down-weights
> easy, already-confident samples and keeps training focused on the confused ones. For a
> balanced two-class task, plain `CrossEntropyLoss` would be a defensible simpler choice.

## 🎓 Training Your Own Model

### 1. Prepare Your Data
Update the data paths in the `Config` class within [`config.py`](config.py):

```python
@dataclass
class Config:
    data_dir: str = "/path/to/your/face-mask-dataset"
    test_data_dir: str = "/path/to/your/secondary-test-dataset"
    # ... other configurations
```

`data_dir` must contain `Train/`, `Validation/` and `Test/` folders; `test_data_dir` is an
optional second dataset evaluated after training (it is skipped if the path is absent).
The Kaggle defaults are absolute Colab/Kaggle paths, so you will almost certainly need to
override them locally.

### 2. Start Training
```bash
python train.py
```

The training process includes:
- ✨ **Data Augmentation**: Rotation, flipping, color jittering for better generalization
- 🛑 **Early Stopping**: Prevents overfitting (patience: 7 epochs)
- 📉 **Learning Rate Scheduling**: Adaptive learning rate adjustment
- 💾 **Model Checkpointing**: Automatic saving of best performing model
- 🎲 **Seeded runs**: `torch`/`numpy`/`random` are seeded from `Config.seed` (42)

### 3. Monitor Training
The script automatically:
- Evaluates on validation set after each epoch
- Saves the best model based on validation accuracy to [`models/best_mobilenet_mask_detector.pt`](models/best_mobilenet_mask_detector.pt)
- Writes `training_curves.png`, `cm_validation.png`, `cm_test.png` and, when the
  secondary dataset is available, `cm_secondary.png`
- Provides detailed training logs and classification reports

## ⚙️ Configuration

Key hyperparameters (modifiable in [`config.py`](config.py)):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 32 | Training batch size |
| Learning Rate | 1e-4 | Initial learning rate |
| Epochs | 25 | Maximum training epochs |
| Dropout Rate | 0.5 | Dropout probability |
| Weight Decay | 1e-4 | L2 regularization |
| Early Stopping Patience | 7 | Epochs to wait before stopping |
| Seed | 42 | RNG seed for reproducible runs |
| Focal Loss Alpha | 1.0 | Class weighting parameter (uniform on balanced data) |
| Focal Loss Gamma | 2.0 | Focusing parameter |

## 🛠️ Development

### Explore the Training Process
Check out the detailed analysis in [`notebook/Mask Detection.ipynb`](notebook/Mask%20Detection.ipynb) for:
- Data exploration and visualization
- Model architecture details
- Training metrics and plots
- Performance analysis

### Dependencies
Dependencies are declared in [`pyproject.toml`](pyproject.toml) and pinned in
[`uv.lock`](uv.lock). Core packages:
- **PyTorch** 2.7+: Deep learning framework
- **torchvision**: Computer vision utilities
- **OpenCV**: Image and video processing
- **Gradio**: Web interface framework
- **FastAPI**: API framework
- **NumPy**: Numerical computing
- **Pillow**: Image processing
- **matplotlib** / **scikit-learn**: training plots and metrics

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
