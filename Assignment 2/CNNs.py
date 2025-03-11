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
      #######################
      # Start YOUR CODE    #
      #######################
      # number of filter's list for each expanding and respecting contracting layer
      c = [16, 32, 64, 128]

      # first convolution layer receiving the image
      # encoder layers

      # decoder layers

      # last layer returning the output
      #######################
      # END OF YOUR CODE    #
      #######################
  def forward(self,x):
      #######################
      # Start YOUR CODE    #
      #######################
      # encoder

      # decoder

      #######################
      # END OF YOUR CODE    #
      #######################
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
