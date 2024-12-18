# Detection of Adversarial Examples Using Self-Attention Based Convolutional Neural Network
This repository contains the Pytorch implementation for the paper and the dataset for training our proposed model, as presented in the paper *"Detection of Adversarial Examples Using Self-Attention Based Convolutional Neural Network"*, to detect benign and adversarial images.

# Table of Contents
- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Visualizing Results](#visualizing-results)


# Prerequisites
The following information outlines the environment required to run this project:

### System
- **Platform**: Linux/ Windows

### Python
- **Version**: Python 3.9 or above

### Frameworks and Libraries
- **PyTorch**: 1.10.0 or later
- **Torchvision**: 0.11.0 or later
- **Matplotlib**: 3.4.3 or later
- **Pillow**: 8.4.0 or later

# Dataset
Please click the link below to access the dataset:

[Google Drive Dataset](https://drive.google.com/drive/folders/1wf1fZ0X9ti1ztGCpQs2JrmgKTrJ0fZWL?usp=sharing)



# Training the Model

## Guidelines for Training the Model (`train.py`)
### 1. **Dataset Preparation**  
   - Refer to the **Dataset** section for detailed instructions on dataset preparation.  
   - Ensure the dataset is preprocessed and structured correctly as per the requirements outlined.

### 2. **Model Initialization**  
   - The model architecture, including any custom layers and blocks, must be defined in the script.
   - The loss function and optimizer are to be chosen according to the task.
   - Hyperparameters such as learning rate, batch size, and number of epochs should be set in the script.

### 3. **Training Loop**  
   - Training is conducted for a defined number of epochs.
   - During each epoch, the model is trained using the training data, and the weights are updated based on the loss.
   - The loss should be calculated using the output from the model and the ground truth labels.

### 4. **Validation**  
   - The model's performance is evaluated using the validation set after each epoch.
   - Metrics like accuracy, precision, recall, or others can be used to monitor the model’s progress.

### 5. **Model Saving**  
   - Once training is complete, the model's weights should be saved in a specified directory for future use (e.g., for inference or further fine-tuning).

### How to Run:

To start training, execute the following command in the terminal:

```bash
python train.py
```

# Evaluating the Model

## Guidelines for Model Evaluation (`result.py`)

### 1. **Dataset Preparation**  
   - Ensure the test dataset is prepared and follows the same preprocessing steps as the training dataset.  
   - Refer to the **Dataset** section for details on data preparation.

### 2. **Model Loading**  
   - The trained model's weights must be loaded from the saved checkpoint.
   - Verify that the model architecture matches the one used during training.

### 3. **Evaluation Metrics**  
   - Define evaluation metrics such as accuracy, precision, recall, F1 score, or others, depending on the task.
   - Results are computed based on model predictions compared to the ground truth labels.

### 4. **Generate Results**  
   - The script processes the test data and produces predictions using the trained model.
   - Evaluation metrics are calculated and displayed or saved as output.


### How to Run:

To evaluate the model, execute the following command in the terminal:

```bash
python result.py
```

# Visualizing Results
## Guidelines for Visualizing Attention Maps (`visualize_attention_map.py`)

### 1. **Dataset Preparation**  
   - Ensure the input image is preprocessed and follows the format required by the model.  
   

### 2. **Model Loading**  
   - The trained model's weights must be loaded from the saved checkpoint.
   - Confirm the model architecture matches the one used for training and evaluation.

### 3. **Attention Map Visualization**  
   - Attention maps highlight the regions of the image the model focuses on during processing.
   - The script extracts and visualizes attention maps from specified layers of the model.

### 4. **Save or Display Visualization**  
   - Attention maps are displayed using `matplotlib` or saved as image files for further analysis.

### How to Run:

To visualize attention maps, execute the following command in the terminal:

```bash
python visualize_attention_map.py
```


# Acknowledgment
Adversarial images were generated using adversarial attacks implemented via the Foolbox library, available at https://github.com/bethgelab/foolbox.
