import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoStreamNet(nn.Module):
    def __init__(self, net1, net2, param):
        super(TwoStreamNet, self).__init__()
        
        self.out_size = param['szOut']
        
        # Define the input layer
        self.input_layer = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        
        # Extract conv1 layers from net1 and net2
        conv1_1 = net1.features[0]
        conv1_2 = net2.features[0]
        
        # Combine weights of conv1 layers
        weights_ = torch.zeros(conv1_1.weight.size(0), conv1_1.weight.size(1) + 1, conv1_1.weight.size(2), conv1_1.weight.size(3))
        weights_[:, :3, :, :] = conv1_1.weight.data
        weights_[:, 3:, :, :] = conv1_2.weight.data
        
        # Initialize combined conv1 layer
        self.skip_conv1 = nn.Conv2d(4, conv1_1.weight.size(0), kernel_size=conv1_1.kernel_size, stride=3, padding=2)
        self.skip_conv1.weight = nn.Parameter(weights_)
        self.skip_conv1.bias = nn.Parameter(conv1_1.bias.data)
        
        # Clear variables
        del conv1_1, conv1_2, weights_
        
        # Combine layers from net1 and net2
        self.layers = nn.Sequential(
            self.skip_conv1,
            *list(net1.features[1:-3]),
            *list(net2.features[1:-3])
        )
        
        # Rename layers
        self.rename_layers()
    
    def rename_layers(self):
        layer_names = [
            'relu1_1', 'norm1_1', 'pool1_1', 'conv2_1', 'relu2_1', 'norm2_1', 'pool2_1',
            'conv3_1', 'relu3_1', 'conv4_1', 'relu4_1', 'conv5_1', 'relu5_1', 'pool5_1',
            'fc6_1', 'relu6_1', 'drop6_1', 'fc7_1', 'relu7_1', 'drop7_1', 'fc8_1', 'relu8_1', 'drop8_1', 'fc9_1',
            'relu1_2', 'norm1_2', 'pool1_2', 'conv2_2', 'relu2_2', 'norm2_2', 'pool2_2', 'conv3_2', 'relu3_2'
        ]
        
        for i, layer in enumerate(self.layers):
            layer.name = layer_names[i]
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.layers(x)
        return x

# Example usage
# Assuming net1 and net2 are pre-trained models and param is a dictionary with required parameters
# net1 = models.vgg16(pretrained=True)
# net2 = models.vgg16(pretrained=True)
# param = {'szOut': 1000, 'DataAugmentation': None, 'Normalization': 'none'}
# model = TwoStreamNet(net1, net2, param)