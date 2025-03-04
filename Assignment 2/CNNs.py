import torch.nn as nn
import torch
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F


class SimpleConvNet(pl.LightningModule):
    def __init__(self, in_channels=3, num_classes=1, dropout_rate=0, conv_channels=[16, 32], linear_features=[60]):
        super().__init__()
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
        x = self.layers(x)
        x = self.classifier(x)
        return x


class UNet(pl.LightningModule):
    def __init__(self, n_classes=1, in_ch=3):
        super().__init__()
        #######################
        # Start YOUR CODE    #
        #######################
        # number of filter's list for each expanding and respecting contracting layer
        c = [16, 32, 64, 128]


        self.encoder = nn.Sequential(
            # first convolution layer receiving the image
            nn.Conv2d(in_channels=in_ch, out_channels=c[0], kernel_size=3, padding=1),
            nn.ReLU(),
            # encoder layers
            nn.Conv2d(in_channels=c[0], out_channels=c[1], kernel_size=3, padding=1),
            nn.Conv2d(in_channels=c[1], out_channels=c[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=c[1], out_channels=c[2], kernel_size=3, padding=1),
            nn.Conv2d(in_channels=c[2], out_channels=c[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=c[2], out_channels=c[3], kernel_size=3, padding=1),
            nn.Conv2d(in_channels=c[3], out_channels=c[3], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.decoder = nn.Sequential(
            # decoder layers
            nn.Conv2d(in_channels=c[3], out_channels=c[2], kernel_size=3, padding=1),
            nn.Conv2d(in_channels=c[2], out_channels=c[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=c[2], out_channels=c[1], kernel_size=3, padding=1),
            nn.Conv2d(in_channels=c[1], out_channels=c[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=c[1], out_channels=c[0], kernel_size=3, padding=1),
            nn.Conv2d(in_channels=c[0], out_channels=c[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # last layer returning the output
            nn.Conv2d(in_channels=c[0], out_channels=n_classes, kernel_size=1)
            )
        #######################
        # END OF YOUR CODE    #
        #######################
    def forward(self,x):
        #######################
        # Start YOUR CODE    #
        #######################
        # encoder
        x = self.encoder(x)    
        # decoder
        x = self.decoder(x)
        #######################
        # END OF YOUR CODE    #
        #######################
        return x


def conv3x3_bn(ci, co):
    #######################
    # Start YOUR CODE    #
    #######################
    return nn.Conv2d(in_channels=ci, out_channels=co, kernel_size=3, padding=1),
    #######################
    # end YOUR CODE    #
    #######################

def encoder_conv(ci, co):
    #######################
    # Start YOUR CODE    #
    #######################
    pass
    #######################
    # end YOUR CODE    #
    #######################

class deconv(nn.Module):
  def __init__(self, ci, co):
    super(deconv, self).__init__()
    #######################
    # Start YOUR CODE    #
    #######################
    pass
    #######################
    # end YOUR CODE    #
    #######################

  def forward(self, x1, x2):
      #######################
      # Start YOUR CODE    #
      #######################
      x=x1
      #######################
      # end YOUR CODE    #
      #######################
      return x
