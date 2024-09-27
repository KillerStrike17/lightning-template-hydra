# README for Lightning-Template-Hydra

## Project Overview

The **Lightning-Template-Hydra** project is designed to provide a flexible and efficient pipeline for training, evaluating, and inferring models. This project includes implementations for both **Cat-Dog Classifier** and **Dog Breed Classifier**. The overall pipeline is built with **PyTorch Lightning**, and the training, evaluation, and inference processes can be run in Docker environments with volume mounts for data persistence.

This project leverages the following key technologies:
- **PyTorch Lightning** for easy model development and management.
- **Hydra** for managing configurations.
- **Docker** and **Docker Compose** for containerizing the pipeline.
- **Volume Mounts** to ensure all outputs (logs, models, and predictions) are saved persistently.

---

## Directory Structure

The following is the core structure of the project:

```
.
├── data
│   ├── cat_dog              # Dataset containing cat anddogs
│   ├── dog_breeds           # Train/validation split for dog breeds
├── logs                     # Logs for training, evaluation, and inference
├── output                   # Output folder for storing predictions
├── samples                  # Placeholder for sample data used for testing
└── src
    ├── datamodules          # Data loading and augmentation scripts
    ├── models               # Cat-Dog and Dog-Breed classifier models
    ├── train.py             # Training script
    ├── eval.py              # Evaluation script
    └── infer.py             # Inference script
```

---

## How to Set Up the Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/megatron17/lightning-template-hydra.git
   cd lightning-template-hydra
   ```

2. **Ensure Docker is Installed:**
   The project is designed to run in a Docker environment. Please ensure Docker is installed on your machine.

3. **Build the Docker Image:**
   You can build the Docker image for this project by running the following command:
   ```bash
   docker-compose build
   ```

---

Here's a detailed explanation of the **Training**, **Evaluating**, and **Inferencing** scripts, along with the **data loading mechanism** used in the project:

---

## Training, Evaluating, and Inferencing Scripts

### 1. **Training Script (`train.py`)**

The `train.py` script is designed to train either the Cat-Dog Classifier or the Dog Breed Classifier, depending on the selected model. It utilizes **PyTorch Lightning** for streamlined training and handling the training loop, validation, and logging.

Key Features:
- The model and dataset are loaded based on user configurations.
- Logging is handled using **Rich** and **Tensorboard** for visualizing training progress, losses, and metrics.
- The **DataLoader** splits the dataset into training and validation sets, applying appropriate data augmentations.
- The script saves checkpoints during training to the output folder using Lightning's inbuilt checkpointing mechanism.

You can run the training script using Docker:

```bash
docker-compose up train
```

---

### 2. **Evaluation Script (`eval.py`)**

The `eval.py` script evaluates the trained model on the test dataset to measure its performance. It loads the best checkpoint (saved during training) and performs predictions on the test data to calculate various metrics like accuracy, precision, recall, etc.

Key Features:
- Loads the saved model checkpoint.
- Uses test DataLoader to evaluate the model on the test dataset.
- Saves evaluation logs in the logs directory for easy tracking.
  
You can run the evaluation script using Docker:

```bash
docker-compose up evaluate
```

---

### 3. **Inference Script (`infer.py`)**

The `infer.py` script performs inference (predictions) on new data/images. It loads a pre-trained model and processes a folder of input images, predicting their labels. The predictions are saved as images with their labels and confidence scores.

Key Features:
- It supports two classifiers: **Cat-Dog** and **Dog Breed**.
- Images are preprocessed and transformed into tensors suitable for model input.
- The script saves the results in the specified output folder.

You can run the inference script using Docker:

```bash
docker-compose up infer
```

---

## Data Loading Mechanism

The data loading mechanism is handled using **PyTorch's DataLoader** class, which efficiently loads and preprocesses the dataset. The mechanism is as follows:

### Data Setup
The project assumes that you have structured datasets stored in the `data` directory. This dataset includes folders with images for different classes (dogs and cats or dog breeds). 

### Data Augmentation and Preprocessing
The data loading is facilitated by the **`torchvision.transforms`** module. The images are preprocessed as follows:
1. **Resize**: Resized to a standard shape (224x224).
2. **Normalization**: Normalizes image pixel values based on predefined mean and standard deviation.
3. **Random Augmentations** (during training): To enhance the dataset diversity, random transformations (like flips, crops) are applied.

### DataLoader Usage
The **DataLoader** splits the dataset into batches for both training and validation:
- **Training DataLoader**: Loads the training data with augmentations, shuffled before every epoch.
- **Validation/Test DataLoader**: Loads the validation or test data without shuffling but still applies resizing and normalization.

The `train.py`, `eval.py`, and `infer.py` scripts all use this DataLoader mechanism to ensure the dataset is properly prepared for training, evaluation, and inference.

### Example of DataLoader

Here's an example snippet from the training process:
```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='data/dog_breeds/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = datasets.ImageFolder(root='data/dog_breeds/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

- **ImageFolder**: Automatically maps each folder of images to a class label based on the folder name.
- **Transforms**: Resize, convert to tensor, and normalize the images.
- **Batch Size**: You can modify this based on the memory capacity of your system or the available GPUs.

---

### Volume Mounts in Docker

The volume mounts ensure that the input data, logs, and output results are available on your local machine, outside the Docker containers. This allows for easy access to training outputs, checkpoints, logs, and prediction results. When you run the containers:

- **Data** is mounted in the `/data` directory.
- **Logs** are mounted in the `/logs` directory.
- **Outputs** (like predictions and checkpoints) are saved in the `/output` directory.

These volumes ensure persistence across runs, so even if the container stops, your data and results remain intact.

---

This detailed breakdown should help you understand how the scripts work and how to efficiently train, evaluate, and infer using Docker, along with proper data handling mechanisms for any custom datasets.

## Docker Volume Mounts

The project uses volume mounts to ensure that logs, datasets, and outputs are preserved across container runs. Below is the Docker Compose configuration for the service and volume mounts:

```yaml
version: '3.8'

services:
    train:
        build:
            context: .
            dockerfile: Dockerfile
        command: python src/train.py
        volumes:
            - ./logs:/logs          # Logs directory
            - ./data:/data          # Data directory
            - ./output:/output      # Output directory for checkpoints and results

    evaluate:
        build:
            context: .
            dockerfile: Dockerfile
        command: python src/eval.py
        volumes:
            - ./logs:/logs          # Logs directory
            - ./data:/data          # Data directory
            - ./output:/output      # Output directory for evaluation results

    infer:
        build:
            context: .
            dockerfile: Dockerfile
        command: python src/infer.py
        volumes:
            - ./logs:/logs          # Logs directory
            - ./data:/data          # Data directory
            - ./output:/output      # Output directory for prediction results

volumes:
    logs:
    data:
    output:
```

The folders `logs`, `data`, and `output` are mounted on the host machine, which ensures that the data is retained even when the Docker containers are stopped or removed.

---

## Model Classifier

The project includes two classifiers:

1. **Cat-Dog Classifier**: This model classifies images as either a cat or a dog.
2. **Dog Breed Classifier**: This model classifies images into 10 different dog breeds:
    - Beagle
    - Boxer
    - Bulldog
    - Dachshund
    - German Shepherd
    - Golden Retriever
    - Labrador Retriever
    - Poodle
    - Rottweiler
    - Yorkshire Terrier

Both classifiers are implemented in the `models/` folder and can be easily loaded using the checkpoints saved during training.

---

## Requirements

All Python dependencies are managed via `pyproject.toml`. If you need to update or install the dependencies, you can do so by ensuring that `pip` installs from this file:

```bash
pip install .
```

Alternatively, you can use the converted `requirements.txt` to install dependencies in your local environment.

```bash
pip install -r requirements.txt
```