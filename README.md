
# AE-UNet: An Efficient and Privacy-Preserving Biomedical Image Segmentation Framework

This repository contains the official PyTorch implementation for the paper: **"An Efficient and Privacy-Preserving Biomedical Image Segmentation Framework for Remote Patient Diagnosis"**.

Our work introduces **Attention-Enhanced UNet (AE-UNet)**, a novel framework designed for accurate and privacy-conscious medical image segmentation on IoMT devices. The framework accurately segments lesion areas for diagnosis while enabling the obscuring of non-lesional, personally identifiable regions.

![Framework of the proposed AES](https://i.imgur.com/your-image-link-here.png)
*Fig. 1. The Framework of the proposed AES. (You can replace the link above with a link to your framework image)*

## Key Innovations

1.  **Outlook Attention Module (OAM)**: Augments the Swin Transformer's ability to capture fine-grained variations within diseased tissues by modeling local relationships explicitly.
2.  **Criss-cross Feature Compression Module (CFCM)**: Replaces standard patch merging in the Swin Transformer to better preserve detailed structural information and improve the segmentation of small-scale objects.
3.  **Privacy-by-Design**: The model's high accuracy allows for the effective isolation of pathological regions, which can then be used to blur or redact the rest of the image, thus preserving patient privacy.

## Setup

### Prerequisites

-   Python 3.8+
-   PyTorch 1.9+
-   CUDA 11.1+ (for GPU support)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/AE-UNet.git
    cd AE-UNet
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment.** This command depends on your operating system and terminal:

| Operating System / Terminal | Activation Command                               |
| :-------------------------- | :----------------------------------------------- |
| **Windows (PowerShell)**    | `.\venv\Scripts\Activate.ps1`                    |
| **Windows (Command Prompt)**| `venv\Scripts\activate.bat`                      |
| **macOS / Linux (bash/zsh)**| `source venv/bin/activate`                       |

    > **Note for PowerShell users:** If you get an error about script execution being disabled, run the following command to allow scripts for the current session, then try activating again:
    > `Set-ExecutionPolicy RemoteSigned -Scope Process`

4.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Datasets

The model was evaluated on the following public datasets. Please download them and organize them as described below.

-   **ISIC 2018 (Task 1)**: [Download Link](https://challenge.isic-archive.com/data/)
-   **Kvasir-SEG**: [Download Link](https://datasets.simula.no/kvasir-seg/)
-   **CVC-ClinicDB**: [Download Link](https://polyp.grand-challenge.org/CVCClinicDB/)

### Directory Structure

Place the datasets in a `data` directory with the following structure. The `train.py` script expects these exact folder names.

```
./data/
├── isic2018/
│   ├── ISIC2018_Task1-2_Training_Input/
│   └── ISIC2018_Task1_Training_GroundTruth/
├── kvasir/
│   ├── images/
│   └── masks/
└── clinicdb/
    ├── Original/
    └── Ground Truth/
```

## Usage

### Training

The main training script `train.py` handles model training and validation.

To train the AE-UNet model on the ISIC 2018 dataset:

```bash
python train.py --dataset isic2018 --data_path ./data/isic2018 --epochs 120 --batch_size 8 --lr 1e-3
```

**Key Arguments:**
-   `--dataset`: Name of the dataset (`isic2018`, `kvasir`, `clinicdb`).
-   `--data_path`: Path to the root directory of the dataset.
-   `--epochs`: Number of training epochs.
-   `--batch_size`: Training batch size. Adjust based on your GPU memory.
-   `--lr`: Initial learning rate.
-   `--model_name`: Name for the saved model checkpoints.

Checkpoints and logs will be saved in the `checkpoints/` directory.

### Evaluation

Use the `evaluate.py` script to test a trained model on the test set.

```bash
python evaluate.py --dataset isic2018 --data_path ./data/isic2018 --checkpoint_path ./checkpoints/ae_unet_isic2018_best.pth
```

**Key Arguments:**
-   `--checkpoint_path`: Path to the trained model `.pth` file.
-   `--data_path`: Path to the dataset for evaluation.
-   `--save_path`: Directory to save segmentation mask predictions.

The script will print the evaluation metrics (mIoU, F1-score, Precision, Recall) and save the predicted masks to the specified directory.

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{yang2024aeunet,
  title={An Efficient and Privacy-Preserving Biomedical Image Segmentation Framework for Remote Patient Diagnosis},
  author={Yang, Bin and Liu, Yunxing and Cao, Enguo},
  journal={Journal of Electronic Imaging},
  year={2024}
}
```

## License

This project is licensed under the MIT License. 
