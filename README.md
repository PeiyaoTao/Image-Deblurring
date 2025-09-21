# Image Deblurring with U-Net and Vision Transformer

A PyTorch project to implement, train, and compare a U-Net and a Vision Transformer (ViT) for single-image deblurring.

![Result Comparison](https://i.imgur.com/your-image-placeholder.png)
*(Placeholder: Replace with a URL to your best result image)*

---

## Overview

This project provides a complete workflow for tackling image deblurring using modern deep learning architectures. It includes scripts for synthetic data generation, a structured Jupyter Notebook for training and evaluation, and a comparison of a state-of-the-art CNN (U-Net with a ResNet50 backbone) against a Transformer-based model (ViT). The goal is to restore sharp, high-quality images from blurry inputs and evaluate which architecture performs better on this image-to-image translation task.

---

## Features

- **Dual Architecture Comparison:** Implements both a U-Net and a ViT for a direct performance comparison.
- **Automated Data Setup:** The notebook automatically downloads the DIV2K dataset if it's not found locally.
- **Synthetic Data Generation:** Includes logic to generate a diverse dataset with motion, defocus, and Gaussian blurs.
- **Structured Workflow:** A self-contained Jupyter Notebook (`deblurring.ipynb`) handles the entire pipeline from data preparation to evaluation.
- **Advanced U-Net Decoder:** Utilizes batch normalization and Squeeze-and-Excitation (scSE) attention in the U-Net decoder to improve quality.
- **Efficient Fine-Tuning:** Implements a partial fine-tuning strategy for the ViT to manage computational cost.
- **Robust Evaluation:** Evaluates models on both 224x224 crops and full-resolution images using a sliding-window inference method.
- **Metrics:** Calculates Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM).

---

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a Conda Environment:**
    ```bash
    conda create -n deblur python=3.9
    conda activate deblur
    ```

3.  **Install PyTorch with CUDA support:**
    (Visit the [PyTorch website](https://pytorch.org/get-started/locally/) to get the command for your specific CUDA version)
    ```bash
    # Example for CUDA 12.1
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```

4.  **Install project dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage Workflow

The entire project is managed through the `deblurring.ipynb` Jupyter Notebook. The cells are designed to be run sequentially.

1.  **Run Data Download & Setup Cells:** Execute the initial data preparation cells. The notebook will:
    - **Automatically download** the DIV2K dataset if it's not found.
    - Organize the images into `train` and `test` sets.
    - Generate synthetic blurry images.
    - Create a smaller, cropped dataset for efficient training.
    - Create a `validation` set from the training data.

2.  **Train Models:** Run the training cell. This will instantiate, create optimizers for, and train both the U-Net and ViT models sequentially. Final weights will be saved as `.pth` files.

3.  **Evaluate Models:** Run the final evaluation cells to see a qualitative and quantitative comparison of the trained models on both random cropped images and random full-resolution images.

---

## File Structure

```
.
├── deblurring.ipynb         # Main Jupyter Notebook for training and evaluation
├── README.md                # This README file
├── requirements.txt         # Project dependencies
└── DIV2K_train_HR/          # (Auto-generated) Original data directory
├── train/
│   ├── sharp/
│   └── blur/
└── test/
├── sharp/
└── blur/
└── cropped_dataset/         # (Auto-generated) Cropped data directory
├── train/
├── validation/
└── test/
```

---

## `requirements.txt`

```
numpy
matplotlib
opencv-python
scikit-image
tqdm
requests
segmentation-models-pytorch
transformers
timm
einops
```
