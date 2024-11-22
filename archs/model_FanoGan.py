import numpy as np
import torch.nn as nn



class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        
        # Add more channels in middle layers
        self.model = nn.Sequential(
            nn.ConvTranspose2d(opt.latent_dim, 1024, 4, 1, 0, bias=False),  # Increased from 512
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1)
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        # Split the model into features and classification parts
        self.features = nn.Sequential(
            # 64x64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 16x16
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 8x8
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 4x4
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifier = nn.Conv2d(512, 1, 4, 1, 0, bias=False)  # 1x1

    def forward(self, img):
        features = self.features(img)
        validity = self.classifier(features)
        return validity.view(-1, 1)
    
    def forward_features(self, img):
        """Return intermediate features for feature matching loss"""
        return self.features(img)


"""class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        
        self.model = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(opt.channels, 64, 4, stride=2, padding=1, bias=False),  # 64 x 32 x 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),  # 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),  # 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),  # 512 x 4 x 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=False),  # 1024 x 2 x 2
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25)
        )
        
        # Calculate the size of flattened features
        ds_size = opt.img_size // (2 ** 5)  # Five downsampling operations
        self.flatten_size = 1024 * ds_size * ds_size
        
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            
            nn.Linear(512, opt.latent_dim),
            nn.Tanh()
        )

    def forward(self, img):
        features = self.model(img)
        features = features.view(features.size(0), -1)
        latent_vector = self.fc(features)
        return latent_vector

"""



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Only create skip connection if channels change
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
            
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv_block(x)
        out = out + identity
        out = self.relu(out)
        if self.downsample:
            out = self.downsample(out)
        return out


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25)
        )
        
        self.layers = nn.ModuleList([
            ResBlock(64, 128, nn.AvgPool2d(2)),
            ResBlock(128, 256, nn.AvgPool2d(2)),
            ResBlock(256, 512, nn.AvgPool2d(2)),
            ResBlock(512, 512)
        ])
        
        # Calculate final feature map size
        ds_size = opt.img_size // 16  # Divided by 2^4 due to 4 downsampling operations
        flatten_size = 512 * ds_size * ds_size
        
        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Linear(512, opt.latent_dim),
            nn.Tanh()
        )

    def forward(self, img):
        out = self.initial(img)
        for i, layer in enumerate(self.layers):
            out = layer(out)
        return self.final(out)