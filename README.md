# Detection of Adversarial Examples Using Self-Attention Based Convolutional Neural Network
This repository contains the Pytorch implementation for the paper and the dataset for training our proposed model, as presented in the paper *"Detection of Adversarial Examples Using Self-Attention Based Convolutional Neural Network"*, to detect benign and adversarial images.

# Abstract
Deep neural networks (DNNs) excel in image classification but are vulnerable to adversarial attacks, where subtle input changes can lead to misclassifications, raising security concerns in critical fields. One method to counter adversarial attacks is categorizing input images as adversarial or benign. Accurately identifying adversarial images with minimal perturbations is challenging due to nearly imperceptible differences and existing works have yet to address the classification of benign and adversarial images for varying perturbation levels. Additionally, existing techniques often struggle to generalize to unseen attack types, which is crucial for real-world applications, as evolving adversarial attacks leave DNNs susceptible to new threats.
To address these challenges, this study proposes a classification model using a scratch Convolutional Neural Network (CNN) with modified self-attention block. This model is designed to distinguish subtle alterations caused by adversarial attacks at different perturbation levels and to generalize in classifying benign and adversarial images of previously unseen attacks, ensuring the model handles new adversarial attacks without retraining. Due to the lack of benchmark datasets, we created a large and diverse adversarial image dataset derived from popular image datasets, incorporating adversarial images from three different attacks, each with three perturbation levels. Our proposed model achieves an impressive accuracy of 97\% to 98\% and approaches a nearly perfect AUC score. It maintains a low false positive rate of 2\% to 5\%, crucial for minimizing misclassifications in adversarial attack detection. Additionally, our proposed model outperforms existing state-of-the-art techniques and possesses 13.04 GFLOPs, 6.52 GMACs, and a rapid inference time of 0.2 seconds, ensuring efficiency and responsiveness for practical applications.

# Table of Contents
- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Visualizing Results](#visualizing-results)


# Prerequisites
The following information outlines the environment required to run this project:

### System
- **Platform**: Linux

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

### 1. **Dataset Preparation**  
   - Refer to the **Dataset** section for detailed instructions on dataset preparation.  
   - Ensure the dataset is preprocessed and structured correctly as per the requirements outlined.

### 2. **Model Initialization**  
   - The model architecture, including any custom layers and blocks, must be defined in the script.
   - The loss function and optimizer (e.g., Adam, SGD) are to be chosen according to the task.
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



