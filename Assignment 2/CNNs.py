import torch.nn as nn
import torch
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F


class SimpleConvNet(pl.LightningModule):
    def __init__(self, in_channels=3, num_classes=1, dropout_rate=0, conv_channels=[16, 32], linear_features=[60], use_residuals=False):
        super().__init__()
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.use_residuals = use_residuals

        if self.use_residuals:
            self.block1= nn.Sequential(
                # conv block 1
                nn.Conv2d(in_channels=in_channels, out_channels=conv_channels[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(conv_channels[0]),
                nn.ReLU(),
                nn.MaxPool2d(2, 2))

            # Res block
            self.res_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(conv_channels[i], conv_channels[i + 1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(conv_channels[i + 1]),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2)
                ) for i in range(len(conv_channels) - 1)
            ])
                    
            # dropout
            self.dropout = nn.Dropout(dropout_rate)
            

            self.classifier = nn.Sequential(
                # linear layers
                nn.AdaptiveAvgPool2d(output_size=(4, 4)),
                nn.Flatten(),
                nn.Linear(in_features=4 * 4 * conv_channels[-1], out_features=linear_features[0]),
                nn.ReLU(),
                nn.Linear(in_features=linear_features[0], out_features=num_classes)
            )
        else:
            #######################
            # PUT YOUR CODE HERE  #
            #######################
            self.layers = nn.Sequential(
                # conv block 1
                nn.Conv2d(in_channels=in_channels, out_channels=conv_channels[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(conv_channels[0]),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                # conv block 2 to n
                *[
                    nn.Sequential(
                        nn.Conv2d(in_channels=conv_channels[i], out_channels=conv_channels[i+1], kernel_size=3, padding=1),
                        nn.BatchNorm2d(conv_channels[i+1]),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2)
                    ) for i in range(len(conv_channels) - 1)
                ],
                    
                # dropout
                nn.Dropout(dropout_rate)
            )

            self.classifier = nn.Sequential(
                # linear layers
                nn.AdaptiveAvgPool2d(output_size=(4, 4)),
                nn.Flatten(),
                nn.Linear(in_features=4 * 4 * conv_channels[-1], out_features=linear_features[0]),
                nn.ReLU(),
                nn.Linear(in_features=linear_features[0], out_features=num_classes)
            )
            #######################
            # END OF YOUR CODE    #
            #######################

    def forward(self, x):
        if self.use_residuals:
            x = self.block1(x)
            for res_block in self.res_blocks:
                residual = x  # Save the input
                x = res_block(x)
                if residual.shape == x.shape:  # Add skip connection only if dimensions match
                    x += residual
            x = self.dropout(x)
            x = self.classifier(x)
        else:   
            x = self.layers(x)
            x = self.classifier(x)
        return x


class UNet(pl.LightningModule):
    def __init__(self, n_classes=1, in_ch=3):
        super().__init__()
        c = [16, 32, 64, 128]

        self.conv1 = encoder_conv(in_ch, c[0])
        self.conv2 = encoder_conv(c[0], c[1])
        self.conv3 = encoder_conv(c[1], c[2])
        self.conv4 = encoder_conv(c[2], c[3])

        self.bottleneck = nn.Sequential(
            conv3x3_bn(c[3], c[3]),
            conv3x3_bn(c[3], c[3])
        )

        self.upconv4 = deconv(c[3], c[2])
        self.upconv3 = deconv(c[2], c[1])
        self.upconv2 = deconv(c[1], c[0])
        self.upconv1 = deconv(c[0], c[0])

        self.last = nn.Conv2d(c[0], n_classes, kernel_size=1)


    def forward(self,x):
        x1 = self.conv1(x) 
        x2 = self.conv2(x1)  
        x3 = self.conv3(x2)  
        x4 = self.conv4(x3) 

        # bottleneck
        x_b = self.bottleneck(x4)  # shape: (batch, 128, H/8, W/8)

        x = self.upconv4(x_b, x3)  
        x = self.upconv3(x, x2) 
        x = self.upconv2(x, x1) 
        x = self.upconv1(x, x) 

        x = self.last(x)  # shape: (batch, n_classes, H, W)

        return x

def conv3x3_bn(ci, co):
    return nn.Sequential(
        nn.Conv2d(in_channels=ci, out_channels=co, kernel_size=3, padding=1),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True)
    )

def encoder_conv(ci, co):
    return nn.Sequential(
        conv3x3_bn(ci, co),
        conv3x3_bn(co, co),
        nn.MaxPool2d(2, 2)
    )

class deconv(nn.Module):
    def __init__(self, ci, co):
        super(deconv, self).__init__()
        self.up = nn.ConvTranspose2d(ci, co, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
                conv3x3_bn(co, co),
                conv3x3_bn(co, co)
        )
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)  
        return self.conv(x)
