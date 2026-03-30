import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class ConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, pool_size: int = 2):
        super().__init__()
      
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(ch_out) 
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        return x

class DroneDetectorMamba(nn.Module):
    def __init__(self, ch_in: int = 1, num_classes: int = 2, n_mels: int = 128, dropout_rate: float = 0.3):
        super().__init__()

        self.ch_in = ch_in
        self.num_classes = num_classes

        self.conv1 = ConvBlock(ch_in, 32, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, padding=1)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, padding=1)
        self.conv4 = ConvBlock(128, 256, kernel_size=3, padding=1)

        self.mel_features = n_mels // 16 
        self.mamba_d_model = 256 * self.mel_features 
      
        self.mamba1 = Mamba(
            d_model=self.mamba_d_model, 
            d_state=16,  
            d_conv=4,    
            expand=2
        ) 
        
        self.mamba2 = Mamba(
            d_model=self.mamba_d_model, 
            d_state=16,  
            d_conv=4,    
            expand=2
        ) 

        self.classifier = nn.Sequential(
            nn.Linear(self.mamba_d_model, 128), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (batch, 1, n_mels, time)
        
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = self.conv3(x)  
        x = self.conv4(x)  # shape: (batch, 256, n_mels//16, time//16)

     
        b, c, mel, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * mel) 

        x = self.mamba1(x)
        x = self.mamba2(x)

        x = x.mean(dim=1) 
        x = self.classifier(x)

        return x
