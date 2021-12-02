"""
AlexNet convolutional neural network example.

Original paper "ImageNet Classification with Deep Convolutional Neural Networks"
by A. Krizhevsky, I. Sutskever and G. Hinton, can be found here: 
https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

The original network contained 8 layers, 5 convolutional and 
3 fully connected.  The input is 150,528 dimensional, with
[253,440-186,624-64,896-64,896-43,264-4096-4096-1000] neurons in each
consecutive layer respectively.

In the original 2012 paper, the authors used two GTX 580s, which only
had 3Gb of memory each, hence the reason for using two.  This resulted in
having to construct two convolutional networks which made things
more complicated, but here we have no reason to do this, so we will
implement the same model on a single GPU.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda
from torchvision import datasets
from configs import *

# use the gpu if it's available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# some information
print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print("Device: {}".format(device))


class AlexNet(torch.nn.Module):
    """

        (a) The input tensor shape is 224x224x3
        (b) The first convolution layer consts of 96 kernels with
            a size of 11x11x3 and a stride of 4.
        (ba)    There is then a response-normalized and pooled
                transformation which occurs here
        (bb)    The pooling uses a 3x3 size with a stride of 2
                (see section 3.4 of the paper)
        (c) The second convolution layer is constructed from 256
            kernels with a size of 5x5x48
        (d) The third layer has 384 kernels of size 3x3x256.
        (e) The fourth layer also has 384 kernels with a size of 3x3x192.
        (f) The fifth convolution layer has 256 kernels with a size of
            3x3x192.
        (g) There are then 3 fully connected layers with sizes 4096,4096
            and 1000.

    """
    def __init__(self,
        configuration,
    ):
        super().__init__()
        # define the architecture
        self.input_channels = configuration.get('input_channels', 3)
        self.conv1_out      = configuration.get('conv1_out', 96)
        self.conv1_kernel   = configuration.get('conv1_kernel', 11)
        self.conv1_pooling  = configuration.get('conv1_pooling', True)

        self.conv2_out      = configuration.get('conv2_out', 256)
        self.conv2_kernel   = configuration.get('conv2_kernel', 5)
        self.conv2_pooling  = configuration.get('conv2_pooling', True)

        self.conv3_out      = configuration.get('conv3_out', 384)
        self.conv3_kernel   = configuration.get('conv3_kernel', 3)
        self.conv4_out      = configuration.get('conv4_out', 384)
        self.conv4_kernel   = configuration.get('conv4_kernel', 3)

        self.conv5_out      = configuration.get('conv5_out', 256)
        self.conv5_kernel   = configuration.get('conv5_kernel', 3)
        self.conv5_pooling  = configuration.get('conv5_pooling', True)

        self.full1_out      = configuration.get('full1_out', 4096)
        self.full2_out      = configuration.get('full2_out', 4096)
        self.out_channels   = configuration.get('out_channels', 1000)

        self.loc_resp_norm  = configuration.get('loc_resp_norm', True)
        self.pooling_kernel = configuration.get('pooling_kernel', 3)
        self.pooling_stride = configuration.get('pooling_stride', 2)

        # some necessary layers
        self.relu = nn.ReLU()
        self.max_pooling = nn.MaxPool2d(
            self.pooling_kernel,
            self.pooling_stride
        )
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()

        # define the eight layers
        self.conv1_layer = nn.Conv2d(
            in_channels =   self.input_channels,
            out_channels=   self.conv1_out,
            kernel_size =   self.conv1_kernel,
            stride      =   4,
            padding     =   1  
        )
        self.conv2_layer = nn.Conv2d(
            in_channels =   self.conv1_out,
            out_channels=   self.conv2_out,
            kernel_size =   self.conv2_kernel,
            stride      =   4,
            padding     =   1
        )
        self.conv3_layer = nn.Conv2d(
            in_channels =   self.conv2_out,
            out_channels=   self.conv3_out,
            kernel_size =   self.conv3_kernel,
            stride      =   4,
            padding     =   1
        )
        self.conv4_layer = nn.Conv2d(
            in_channels =   self.conv3_out,
            out_channels=   self.conv4_out,
            kernel_size =   self.conv4_kernel,
            stride      =   4,
            padding     =   1
        )
        self.conv5_layer = nn.Conv2d(
            in_channels =   self.conv4_out,
            out_channels=   self.conv5_out,
            kernel_size =   self.conv5_kernel,
            stride      =   4,
            padding     =   1
        )
        self.full1_layer = nn.Linear(
            in_features =   self.conv5_out,
            out_features=   self.full1_out
        )
        self.full2_layer = nn.Linear(
            in_features =   self.full1_out,
            out_features=   self.full2_out,
        )
        self.output_layer = nn.Linear(
            in_features =   self.full2_out,
            out_features=   self.out_channels,
        )

    def forward(self, x):
        x = x.to(device)
        # first convolution
        x = self.conv1_layer(x)
        x = self.relu(x)
        if self.conv1_pooling:
            x = self.max_pooling(x)
        # second convolution
        x = self.conv2_layer(x)
        x = self.relu(x)
        if self.conv2_pooling:
            x = self.max_pooling(x)
        # third convolution
        x = self.conv3_layer(x)
        x = self.relu(x)
        # fourth convolution
        x = self.conv4_layer(x)
        x = self.relu(x)
        # fifth convolution
        x = self.conv5_layer(x)
        if self.conv5_pooling:
            x = self.max_pooling(x)
        x = self.relu(x)
        # fully connected layers
        x = self.flatten(x)
        x = self.full1_layer(x)
        x = self.relu(x)
        x = self.full2_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x
        
class Model:
    """
    """
    def __init__(self,
        configuration
    ):
        self.dataset = configuration.get("dataset", "cifar10")
        self.learning_rate = configuration.get("learning_rate", 1e-3)
        self.batch_size = configuration.get("batch_size", 128)
        self.alex_net = AlexNet(configuration).cuda()
        if(next(self.alex_net.parameters()).is_cuda):
            print("AlexNet is loaded on the GPU!")
        else:
            print("AlexNet is not loaded on the GPU.")
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.alex_net.parameters(), 
            lr=self.learning_rate
        )
        if self.dataset == "cifar10":
            # create datasets
            self.training_data = datasets.CIFAR10(
                root="data",
                train=True,
                download=True,
                transform=ToTensor()
            )
            self.test_data = datasets.CIFAR10(
                root="data",
                train=False,
                download=True,
                transform=ToTensor()
            )
        else:
            # create datasets
            self.training_data = datasets.ImageNet(
                root="data",
                train=True,
                download=True,
                transform=ToTensor()
            )
            self.test_data = datasets.ImageNet(
                root="data",
                train=False,
                download=True,
                transform=ToTensor()
            )
        # create dataloaders
        self.train_dataloader = DataLoader(
            self.training_data,
            batch_size=self.batch_size
        )
        self.test_dataloader = DataLoader(
            self.test_data,
            batch_size=self.batch_size
        )
        
    def train(self,
        epochs: int
    ) -> None:
        size = len(self.train_dataloader.dataset)
        for epoch in range(epochs):
            for batch, (x,y) in enumerate(self.train_dataloader):
                prediction = self.alex_net(x.to(device))
                loss = self.loss_function(prediction, y.to(device))
                # backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(x)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self):
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in self.test_dataloader:
                prediction = self.alex_net(x.to(device))
                test_loss += self.loss_function(prediction, y.to(device)).item()
                correct += (prediction.argmax(1) == y.to(device)).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def save(self, 
        title:  str=''
    ) -> None:
        torch.save(
            self.alex_net.state_dict(), 
            "models/{}.pth".format(title)
        )
    
    def load(self,
        title:  str=''
    ) -> None:
        self.alex_net.load_state_dict(torch.load("models/{}.pth".format(title)))


if __name__ == "__main__":
    
    alex_net = Model(cifar10_config)
    alex_net.train(100)
    alex_net.test()
    alex_net.save("my_alexnet")