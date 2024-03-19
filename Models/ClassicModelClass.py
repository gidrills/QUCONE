from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

class Net(Module): 
    #init is constructor method in phyton, self represents the instance
    def __init__(self):
        super(Net, self).__init__()
        # Defining layers that are specific of the CNN
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 5, kernel_size=5, stride=3, padding=4),
            BatchNorm2d(5), # A normalization layer (not important to study)
            ReLU(inplace=True), # The activation funcion (Rectified Linear Unit)
            MaxPool2d(kernel_size=2, stride=2), # Pooling layer
            # Defining another 2D convolution layer
            Conv2d(5, 5, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(5),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear=Sequential(
            Linear(125,10),
            ReLU(inplace=True),
            Linear(10, 2)
            )
        # Defining a normal layer of neurons (usually called Linear, Dense, or similar)
        self.linear2 = Sequential(
            Linear(2, 1)
        )

    # Defining the forward pass    
    def forward(self, x):
        # First, applying all the convolutional stuff
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1) # not important
        # Second applying all the linear layers (only one for now)   
        x = self.linear(x)
        x = self.linear2(x)
        return x

def return_classic_model() :  
    classic_model = Net()