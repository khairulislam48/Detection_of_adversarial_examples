
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.optim import AdamW
from torch.optim import SGD
from torch.optim import Adam

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns



class FeatureSqueezingLayer(nn.Module):
    def __init__(self, bit_depth):
        super(FeatureSqueezingLayer, self).__init__()
        self.bit_depth = bit_depth

    def forward(self, x):
        quantized_x = torch.floor(x * (2 ** self.bit_depth)) / (2 ** self.bit_depth)
        return quantized_x

class SimplifiedSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SimplifiedSelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value(x).view(batch_size, -1, height * width)
        x = torch.bmm(value, attention.permute(0, 2, 1))
        x = x.view(batch_size, channels, height, width)
        x = self.gamma * x + x

        feature_squeeze = FeatureSqueezingLayer(bit_depth=8)
        x = feature_squeeze(x)

        return x

class ResidualBlockWithTransformerAttention(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlockWithTransformerAttention, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                        nn.BatchNorm2d(out_channels),
                       
                        nn.SELU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(out_channels))
        self.attention = SimplifiedSelfAttention(out_channels)  # Using simplified self-attention
        self.downsample = downsample
        
        self.selu = nn.SELU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.attention(out) 
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        
        out = self.selu(out)
        return out


class ResNetWithAttention(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNetWithAttention, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                        nn.BatchNorm2d(64),
                        
                        nn.SELU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x





validation_dir = ''


model_path = ''


transform = transforms.Compose([
    transforms.Resize((224, 224)),   # imagenet 224x224 cifar10,cifar100 32x32       
    transforms.ToTensor()
])




validation_dataset = ImageFolder(validation_dir, transform=transform)
val_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False)



model = ResNetWithAttention(ResidualBlockWithTransformerAttention, [3, 4, 6, 3], num_classes=2)

device = torch.device("cuda")

model.load_state_dict(torch.load(model_path,map_location=device))
model.to(device)  # Move the model to the GPU





criterion = nn.CrossEntropyLoss()


true_labels = []
predicted_probs = []

epoch_val_accuracy = 0
epoch_val_loss = 0

val_true_labels = []
val_pred_labels = []


overall_true_positives = 0
overall_false_positives = 0
overall_false_negatives = 0
model.eval()
with tqdm(total=len(val_loader), unit='batch') as val_pbar:
    for j, (val_data, val_label) in enumerate(val_loader):
        val_data = val_data.to(device)
        val_label = val_label.to(device)

        val_output = model(val_data)
        val_loss = criterion(val_output, val_label)
        val_probs = torch.nn.functional.softmax(val_output, dim=1)[:, 1]

        true_labels.extend(val_label.cpu().numpy())
        predicted_probs.extend(val_probs.detach().cpu().numpy())

        val_true_labels.extend(val_label.cpu().numpy())
        val_pred_labels.extend(val_output.argmax(dim=1).cpu().numpy())

        
 
       # val_loss = criterion(val_output, val_label)

        acc = (val_output.argmax(dim=1) == val_label).float().mean()
        epoch_val_accuracy += acc.item()  # Accumulate validation accuracy
        epoch_val_loss += val_loss.item()  # Accumulate validation loss

        val_pbar.set_postfix(loss=val_loss.item(), acc=acc.item())
        val_pbar.update()

epoch_val_accuracy /= len(val_loader)
epoch_val_loss /= len(val_loader)




print(f'Accuracy: {epoch_val_accuracy:.4f}')
print(f'Average Loss: {epoch_val_loss:.4f}')


true_labels = np.array(true_labels)
predicted_probs = np.array(predicted_probs)


fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
roc_auc = roc_auc_score(true_labels, predicted_probs)

print(f'ROC AUC Score: {roc_auc:.4f}')


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()






conf_matrix = confusion_matrix(val_true_labels, val_pred_labels)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Validation Set Confusion Matrix')
plt.show()


#conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)

overall_true_positives += conf_matrix[1, 1]
overall_false_positives += conf_matrix[0, 1]
overall_false_negatives += conf_matrix[1, 0]

overall_precision = overall_true_positives / (overall_true_positives + overall_false_positives)
overall_recall = overall_true_positives / (overall_true_positives + overall_false_negatives)
overall_f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)

# Extract values from the confusion matrix
true_positives = conf_matrix[1, 1]
false_positives = conf_matrix[0, 1]
false_negatives = conf_matrix[1, 0]
true_negatives = conf_matrix[0, 0]


print(f'Overall Precision: {overall_precision:.4f}')
print(f'Overall Recall: {overall_recall:.4f}')
print(f'Overall F1 Score: {overall_f1_score:.4f}')

accuracy = (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)


print(f'Overall Accuracy: {accuracy:.4f}')
