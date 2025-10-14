import torch
import torch.nn as nn

# ---------------- Resnet34 ----------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels


        self.conv1 = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding="same")
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding="same")
        self.bn2 = nn.BatchNorm1d(self.out_channels)
        self.relu2 = nn.ReLU()

        # projection nếu số kênh thay đổi
        if self.in_channels != self.out_channels:
            self.proj = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1)
        else:
            self.proj = None

    def forward(self, x):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        out = self.relu2(out)
        return out

class Resnet34(nn.Module):
    def __init__(self, num_classes=2):
        super(Resnet34,self).__init__()
        self.num_classes = num_classes
        self.conv0 = nn.Conv1d(1, 48, 80, 4)
        self.bn0 = nn.BatchNorm1d(48)
        self.relu = nn.ReLU()
        self.pool0 = nn.MaxPool1d(4)

        self.stage0 = nn.Sequential(ResBlock(48, 48), ResBlock(48, 48))
        self.pool1 = nn.MaxPool1d(4)
        self.stage1 = nn.Sequential(ResBlock(48, 96), ResBlock(96, 96))
        self.pool2 = nn.MaxPool1d(4)
        self.stage2 = nn.Sequential(ResBlock(96, 192), ResBlock(192, 192))
        self.pool3 = nn.MaxPool1d(4)
        self.stage3 = nn.Sequential(ResBlock(192, 384), ResBlock(384, 384))


        self.avgpool = nn.AvgPool1d(1)

    def forward(self, x):
        
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu(out)
        out = self.pool0(out)   #48, 620

        out = self.stage0(out)
        out = self.pool1(out) #48, 155

        out = self.stage1(out)
        out = self.pool2(out) #96, 39

        out = self.stage2(out)
        out = self.pool3(out) #192, 9

        out = self.stage3(out) #384, 9

        out = self.avgpool(out)
        # features = out.mean(dim=2)
        # out = self.fc(features)

        # return out, features
        return out


# ---------------- Projector ----------------
class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Projector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim, affine=False)
        )
    
    def forward(self, x):
        return self.net(x)
    

# ---------------- Predictor ----------------
class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.net(x)
    
# ---------------- Classifier ----------------
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.avgpool = nn.AvgPool1d(1)
        self.fc1 = nn.Linear(384, 128)
        self.fc2 = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = self.avgpool(x)
        out = out.mean(dim=2)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        # print("features: ", features)
        return out

# ---------------- Main SimSiam Model ----------------
class Res34SimSiamNoise(nn.Module):
    def __init__(self, dim=512, pred_dim=128, single_source_mode=False):
        super(Res34SimSiamNoise, self).__init__()
        self.encoder1 = Resnet34(num_classes=pred_dim)
        self.encoder2 = Resnet34(num_classes=pred_dim)
     
        self.single_source_mode = single_source_mode

        prev_dim = 384
        self.projector1 = Projector(prev_dim, prev_dim, dim) # replace fc with Projector
        self.projector2 = Projector(prev_dim, prev_dim, dim) # replace fc with Projector

        self.predictor1 = Predictor(dim, pred_dim)
        self.predictor2 = Predictor(dim, pred_dim)


        self.classification_head = Classifier(128, 1)

    def forward(self, ECG, PPG):
        if not self.single_source_mode:
            features_ECG = self.encoder1(ECG)
            features_PPG = self.encoder1(PPG)
            x_ECG = torch.mean(features_ECG, dim=2) 
            # print("features_ECG: ", features_ECG.shape)
            x_PPG = torch.mean(features_PPG, dim=2) 

            z_ECG = self.projector1(x_ECG)
            z_PPG = self.projector2(x_PPG)
            p_ECG = self.predictor1(z_PPG)
            p_PPG = self.predictor1(z_ECG)


            class_pred1 = self.classification_head(features_ECG)
            class_pred2 = self.classification_head(features_PPG)

            return z_ECG, z_PPG, p_ECG, p_PPG, class_pred1, class_pred2
        
        else:
            if PPG is None:
                features_ECG = self.encoder1(ECG)
                class_pred1 = self.classification_head(features_ECG)
                # print("class_pred1: ", class_pred1)
                return class_pred1
            elif ECG is None:
                features_PPG = self.encoder1(PPG)
                class_pred2 = self.classification_head(features_PPG)
                return class_pred2
        
