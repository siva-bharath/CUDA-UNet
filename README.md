# CUDA-UNet

A PyTorch implementation of U-Net with CUDA acceleration for efficient image segmentation.
Custom CUDA kernels are provided via a PyTorch C++ extension for up to **5×** speed-up.

## Features

- U-Net architecture implementation using PyTorch
- CUDA acceleration for faster training and inference
- Custom CUDA kernels via PyTorch C++ extensions
- Support for custom datasets
- Training and evaluation scripts
- Pre-trained models (coming soon)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CUDA-UNet.git
cd CUDA-UNet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Build the custom CUDA extension (optional but recommended for the fastest training):
```bash
python setup.py install
```

## Project Structure

```
CUDA-UNet/
├── models/          # Model architecture definitions
├── utils/           # Utility functions and helpers
├── data/            # Data loading and preprocessing
├── train.py         # Training script
├── requirements.txt # Project dependencies
└── README.md        # This file
```

## Usage

### Training

```bash
python train.py --data_dir /path/to/dataset --epochs 100 --batch_size 32
```
The training script automatically loads the compiled CUDA extension if it is present.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 