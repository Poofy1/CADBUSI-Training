import numpy as np
import torch.nn as nn




class ResBlockG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsample=True):
        super(ResBlockG, self).__init__()
        
        # Calculate padding to maintain size
        padding = (kernel_size - 1) // 2
        
        # Main branch with larger kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut branch remains 1x1
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        
        # Upsampling
        self.upsample = None
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
            
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out
    
    
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(opt.latent_dim, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        self.res_blocks = nn.Sequential(
            ResBlockG(1024, 512),  # 8x8
            ResBlockG(512, 256),   # 16x16
            ResBlockG(256, 128),   # 32x32
            ResBlockG(128, 64)     # 64x64
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(64, opt.channels, 5, 1, 2),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1)
        out = self.initial(z)
        out = self.res_blocks(out)
        return self.final(out)
    

    
class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        # Split the model into features and classification parts
        self.features = nn.Sequential(
            # 64x64
            nn.Conv2d(opt.channels, 64, 4, 2, 1, bias=False),  # 32x32
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        self.classifier = nn.Conv2d(512, 1, 4, 1, 0, bias=False)  # 1x1

    def forward(self, img):
        features = self.features(img)
        validity = self.classifier(features)
        return validity.view(-1, 1)
    
    def forward_features(self, img):
        return self.features(img)

"""
class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim,
                                128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))

    def forward(self, img):
        features = self.forward_features(img)
        validity = self.adv_layer(features)
        return validity

    def forward_features(self, img):
        features = self.model(img)
        features = features.view(features.shape[0], -1)
        return features
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
            nn.Conv2d(opt.channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(0.25)
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
            #nn.Dropout(0.25),
            nn.Linear(512, opt.latent_dim),
            nn.Tanh()
        )

    def forward(self, img):
        out = self.initial(img)
        for i, layer in enumerate(self.layers):
            out = layer(out)
        return self.final(out)