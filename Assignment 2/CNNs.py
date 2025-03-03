import torch.nn as nn
import torch
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F


class SimpleConvNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.layers = nn.Sequential(
            # conv block 1
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # conv block 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # dropout
            nn.Dropout(0.5)
        )

        self.classifier = nn.Sequential(
            # linear layers
            nn.AdaptiveAvgPool2d(output_size=(4, 4)),
            nn.Flatten(),
            nn.Linear(in_features=4 * 4 * 32, out_features=60),
            nn.ReLU(),
            nn.Linear(in_features=60, out_features=1)
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


        self.layers = nn.Sequential(
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
            nn.MaxPool2d(2, 2),
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
        # decoder
        x = self.layers(x)    

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
