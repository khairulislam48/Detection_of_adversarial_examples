import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.optim import SGD


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






device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = ResNetWithAttention(ResidualBlockWithTransformerAttention, [3,4,6,3], num_classes=2) 



model.to(device)  # Move the model to the GPU


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._load_images()
        self.transform = transform
        
    def _load_images(self):
        images = []
        for cls in self.classes:
            class_path = os.path.join(self.root_dir, cls)
            for image_name in os.listdir(class_path):
                images.append((os.path.join(class_path, image_name), cls))
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, cls = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[cls]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label



transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to match the model's input size for imagenet 224x224 , for cifar10,cifar100 32x32
        transforms.ToTensor(),           # Convert PIL Image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

train_data = CustomDataset('imagenet_train', transform)
val_data = CustomDataset('imagenet_val', transform)


train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=8, shuffle=True)
# Training settings
# batch_size = 64
# epochs = 20
lr = 3e-5

# seed = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 400
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
# momentum = 0.9  
# optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler

model.to(device)





train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []




for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    # Initialize tqdm for the training loop
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='batch') as train_pbar:
        for i, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc.item()  # Accumulate accuracy
            epoch_loss += loss.item()  # Accumulate loss

            # Update tqdm progress bar with current loss and accuracy
            train_pbar.set_postfix(loss=loss.item(), acc=acc.item())
            train_pbar.update()

    # Calculate average accuracy and loss for the epoch
    epoch_accuracy /= len(train_loader)
    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    # Validation loop with tqdm
    epoch_val_accuracy = 0
    epoch_val_loss = 0

    with tqdm(total=len(val_loader), desc=f'Validation {epoch}/{epochs}', unit='batch') as val_pbar:
        for j, (val_data, val_label) in enumerate(val_loader):
            val_data = val_data.to(device)
            val_label = val_label.to(device)

            val_output = model(val_data)
            val_loss = criterion(val_output, val_label)

            acc = (val_output.argmax(dim=1) == val_label).float().mean()
            epoch_val_accuracy += acc.item()  # Accumulate validation accuracy
            epoch_val_loss += val_loss.item()  # Accumulate validation loss

            # Update tqdm progress bar for validation
            val_pbar.set_postfix(loss=val_loss.item(), acc=acc.item())
            val_pbar.update()

    # Calculate average validation accuracy and loss for the epoch
    epoch_val_accuracy /= len(val_loader)
    epoch_val_loss /= len(val_loader)
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_accuracy) 

    # Print and save model after each epoch
    print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")


    model_state_path = f'/home/piash/model_{epoch}.pth'


    torch.save(model.state_dict(), model_state_path)

    # if epoch_val_accuracy >= 0.99:
    #     print(f"Early stopping triggered at epoch {epoch+1} (validation accuracy >= 0.99).")
    #     break


   



# Plot the losses
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')

# Set y-axis limits to be between 0 and 1
plt.ylim(0, 1)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()





# Plot the accuracies

plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


