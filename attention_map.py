import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

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
    

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to match the model's input size
        transforms.ToTensor(),           # Convert PIL Image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension


def visualize_attention(model, input_image):
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(input_image)

    # Get the attention maps from the last layer
    attention_maps = []
    for module in model.modules():
        if isinstance(module, SimplifiedSelfAttention):
            # Compute attention weights from the weights of convolutional layers
            query_weights = module.query.weight.squeeze()  # Shape: (out_channels, in_channels // 8, 1, 1)
            key_weights = module.key.weight.squeeze()  # Shape: (out_channels, in_channels // 8, 1, 1)
            energy = torch.matmul(query_weights, key_weights.permute(1, 0))  # Shape: (out_channels, out_channels)
            attention_map = F.softmax(energy, dim=-1)
            attention_maps.append(attention_map)

    # Visualize the attention maps
    num_maps = len(attention_maps)
    fig, axs = plt.subplots(1, num_maps, figsize=(15, 5))
    for i, attention_map in enumerate(attention_maps):
        axs[i].imshow(attention_map.detach().cpu().numpy())
        axs[i].set_title(f'Attention Map {i+1}')
        axs[i].axis('off')
    plt.show()





model = ResNetWithAttention(ResidualBlockWithTransformerAttention, [3, 4, 6, 3], num_classes=2)


model_path = ""
model.load_state_dict(torch.load(model_path))
# Path to your input image


image_path = ""

# Preprocess the image
input_image = preprocess_image(image_path)

# Visualize attention
visualize_attention(model, input_image)



