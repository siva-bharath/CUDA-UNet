# CUDA-UNet

A PyTorch implementation of U-Net with CUDA acceleration for efficient image segmentation.

## Features

- U-Net architecture implementation using PyTorch
- CUDA acceleration for faster training and inference
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

## Project Structure

```
CUDA-UNet/
├── models/          # Model architecture definitions
├── utils/           # Utility functions and helpers
├── data/            # Data loading and preprocessing
├── train.py         # Training script
├── evaluate.py      # Evaluation script
├── requirements.txt # Project dependencies
└── README.md        # This file
```

## Usage

### Training

```bash
python train.py --data_dir /path/to/dataset --epochs 100 --batch_size 32
```

### Evaluation

```bash
python evaluate.py --model_path /path/to/model --test_dir /path/to/test/data
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 