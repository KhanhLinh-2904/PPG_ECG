from torch import nn
import torch
import torch.nn.functional as F
ECG_INPUT_LENGTH = 2400
OUTPUT_EMBED_DIM = 128  
LAYERS = [3, 4, 6, 3] 
BASE_WIDTH = 64
EXPANSION = 4 
class Bottleneck1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        self.expansion = EXPANSION

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class ResNet50_1D(nn.Module):
    def __init__(self, layers=LAYERS, num_classes=OUTPUT_EMBED_DIM):
        super(ResNet50_1D, self).__init__()
        self.in_channels = BASE_WIDTH

        self.conv1 = nn.Conv1d(1, BASE_WIDTH, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(BASE_WIDTH)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck1D, BASE_WIDTH, layers[0])
        self.layer2 = self._make_layer(Bottleneck1D, BASE_WIDTH * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck1D, BASE_WIDTH * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck1D, BASE_WIDTH * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(BASE_WIDTH * 8 * EXPANSION, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * EXPANSION:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * EXPANSION, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * EXPANSION),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * EXPANSION
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):

        x= self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        f1 = self.layer1(x) 
        f2 = self.layer2(f1) 
        f3 = self.layer3(f2) 
        f4 = self.layer4(f3) 
     
        feature = f4
        x = self.avgpool(feature)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, f4, [f1, f2, f3]
    

if __name__ == '__main__':
    
    model = ResNet50_1D(
        layers=LAYERS,
        num_classes=OUTPUT_EMBED_DIM
    )
    
    # Tạo dữ liệu ECG đầu vào mô phỏng (Batch size = 4)
    # Tín hiệu ECG đơn kênh [Batch, Channels=1, Length]
    dummy_ecg = torch.randn(4, 1, ECG_INPUT_LENGTH)
    
    # Lan truyền thuận
    print(f"Shape of dummy ECG input: {dummy_ecg.shape}")
    
    output_ecg_embedding,_,_ = model(dummy_ecg)
    
    # Kết quả
    print(f"Shape of predicted ECG features output: {output_ecg_embedding.shape}")
    print(f"Expected shape: [4, 128] (Batch size, ECG Embedding Dimension)")