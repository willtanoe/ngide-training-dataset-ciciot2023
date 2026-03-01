# Deep Learning-based IoT Intrusion Detection System (IDS)

This repository contains the core experimental codebase for research on developing an Intrusion Detection System (IDS) specifically designed for Internet of Things (IoT) networks. The models are trained and evaluated using the comprehensive **CICIoT2023** dataset.

This project is part of a Master's thesis research in Electrical Engineering (Network Engineering and Cyber Security) at Telkom University.

## Research Overview

The proliferation of IoT devices has introduced massive attack surfaces. Traditional IDS architectures often fail to detect modern, complex, and high-volume attacks targeting IoT ecosystems. This research explores deep learning architectures to classify network traffic into benign or specific attack categories with high precision and low latency.

## Dataset: CICIoT2023

The experiments utilize the [CICIoT2023 dataset](https://www.unb.ca/cic/datasets/iotdataset-2023.html), which represents a large-scale, modern IoT network environment. 

* **Total features:** 46 network traffic features extracted from pcap files.
* **Classes:** 33 attack categories and 1 benign category (grouped into 8 main attack types for macro-analysis).
* **Challenge:** The raw dataset is extremely massive (hundreds of gigabytes).

## System Environment & Hardware Constraints

The data preprocessing and model training pipelines are optimized for the following local workstation specifications. Memory management (chunking and generator-based loading) is heavily implemented to bypass the 16GB system RAM bottleneck during data ingestion.

* **CPU:** AMD Ryzen 7 5700X
* **GPU:** MSI GAMING X TRIO NVIDIA GeForce RTX 3060 Ti (CUDA Enabled)
* **RAM:** 16GB DDR4 3200 MHz
* **OS:** Windows 11 24h2
* **Frameworks:** Python 3.x, PyTorch, Pandas, Scikit-learn

## Methodology

The pipeline in this repository follows a strict analytical workflow:

1.  **Data Preprocessing:** Handling missing values, feature scaling (Standardization/MinMax), and encoding target labels. Due to memory limits, data is processed in chunks.
2.  **Feature Selection:** Dimensionality reduction to optimize the model for real-time IoT deployment.
3.  **Model Architecture:** Custom Deep Learning models built using PyTorch. 
4.  **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, and Confusion Matrix analysis.

## Repository Structure

* `notebooks/`: Jupyter Notebooks for Exploratory Data Analysis (EDA) and initial prototyping.
* `src/`: Python source code containing modularized functions.
    * `data_loader.py`: Scripts for chunk-based dataset loading and preprocessing.
    * `models.py`: PyTorch neural network architectures.
    * `train.py`: Main training loop with loss optimization.
    * `evaluate.py`: Model inference and metric calculation.
* `requirements.txt`: Python dependencies required to run the environment.

## Installation & Setup

It is highly recommended to run this project inside an isolated Python Virtual Environment to prevent dependency conflicts.

### 1. Clone the Repository
```bash
git clone https://github.com/willtanoe/ids-ciciot2023-dl.git
cd ids-ciciot2023-dl
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
```

### 3. Install GPU-Accelerated PyTorch (Critical Step)
Do not install PyTorch directly from the requirements file if you intend to use hardware acceleration. For Windows users with NVIDIA GPUs (e.g., RTX 3060 Ti), install the CUDA-specific wheels first. Ensure your NVIDIA drivers are up to date.

```bash
# Example for CUDA 12.1 (Adjust according to your local CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Remaining Dependencies
Once PyTorch is firmly established with CUDA support, install the rest of the data manipulation and monitoring modules.

```bash
pip install -r requirements.txt
```

## Running the Verification Script
Before executing any heavy data preprocessing on the CICIoT2023 dataset, run the environment check script to validate your hardware limits (16GB RAM constraints and CUDA availability).

```bash
python src/check_env.py
```