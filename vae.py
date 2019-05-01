# Variational Autoencoder for generating faces similar to frey face dataset
# Based mainly on:
    # This lecture: https://www.youtube.com/watch?v=uaaqyVS9-rM
    # This example: https://github.com/pytorch/examples/tree/master/vae
# Main changes:
    # - Convolutional layers instead of fully connected
    # - MSE loss instead of log loss
    #   beacause MSE was derived in maths in the lecture
    # Changed MNIST to frey face dataset

from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import scipy.io
import os

TRAIN_EPOCHS = 200
BATCH_SIZE = 64
LATENT_DIM = 3

IMG_WIDTH = 20
IMG_HEIGHT = 28
IMG_CHANNELS = 1

# Loads MNIST data - was previously used for training
def get_MNIST_data_loader(batchSize):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data',
                        train = True, 
                        download = True,
                        transform = transforms.ToTensor()),
        batch_size = batchSize, 
        shuffle = True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', 
                        train = False, 
                        download = True,
                        transform = transforms.ToTensor()),
        batch_size = batchSize, 
        shuffle=True
    )

    return (train_loader, test_loader)

# Loads Frey Face dataset
def get_FreyFace_data_loader(batchSize):
    ff = scipy.io.loadmat('data/frey_rawface.mat')
    ff = ff["ff"].T.reshape((-1, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH))
    ff = ff.astype('float32')/255.
    size = len(ff)
    ff = ff[:int(size/batchSize)*batchSize]
    ff_torch = torch.from_numpy(ff)

    testStart = int(0.9 * size)
    ff_train = ff_torch[:testStart]
    ff_test = ff_torch[testStart:]

    train_loader = torch.utils.data.DataLoader(ff_train, batchSize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(ff_test, batchSize, shuffle=True)
    
    return (train_loader, test_loader)


class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()

        convFeatures = (20, 30)
        self.convFeatures = convFeatures
        hiddenNum = 50

        # ENCODER
        # convolute -> maxpool -> convolute -> maxpool -> fully connected x2
        self.encodeConv1 = nn.Conv2d(in_channels=IMG_CHANNELS, 
                               out_channels=convFeatures[0], 
                               kernel_size=3, 
                               padding=True)

        self.encodeMaxPool = nn.MaxPool2d(kernel_size=2, 
                                    return_indices=True)

        self.encodeConv2 = nn.Conv2d(in_channels=convFeatures[0], 
                               out_channels=convFeatures[1], 
                               kernel_size=3, 
                               padding=True)

        self.totalPooledSize = convFeatures[1] * (IMG_WIDTH//4) * (IMG_HEIGHT//4)

        self.encodeFC = nn.Linear(in_features=self.totalPooledSize, 
                                  out_features=hiddenNum)

        self.encodeMeanFC = nn.Linear(in_features=hiddenNum, 
                                      out_features=LATENT_DIM)

        self.encodeVarianceFC = nn.Linear(in_features=hiddenNum, 
                                          out_features=LATENT_DIM)

        # DECODER
        # fully connected x2 -> unpool -> convolute -> unpool -> convolute x2
        self.decodeFC1 = nn.Linear(in_features=LATENT_DIM,
                                   out_features=hiddenNum)

        self.decodeFC2 = nn.Linear(in_features=hiddenNum, 
                                   out_features=self.totalPooledSize)

        self.decodeUnPool1 = nn.MaxUnpool2d(kernel_size=2)

        self.decodeUnConv1 = nn.ConvTranspose2d(in_channels=convFeatures[1], 
                                                out_channels=convFeatures[0], 
                                                kernel_size=3, 
                                                padding=True)

        self.decodeUnPool2 = nn.MaxUnpool2d(kernel_size=2)

        self.decodeUnConv2 = nn.ConvTranspose2d(in_channels=convFeatures[0], 
                                                out_channels=IMG_CHANNELS, 
                                                kernel_size=3, 
                                                padding=True)

        self.decodeFinalConv = nn.Conv2d(in_channels=IMG_CHANNELS, 
                                         out_channels=IMG_CHANNELS, 
                                         kernel_size=3, 
                                         padding=True)

        self.normalDist = torch.distributions.Normal(torch.zeros(LATENT_DIM), torch.ones(LATENT_DIM))

    def encode(self, x):
        x = F.leaky_relu(self.encodeConv1(x))
        (x, self.indices1) = self.encodeMaxPool(x)
        x = F.leaky_relu(self.encodeConv2(x))
        (x, self.indices2) = self.encodeMaxPool(x)
        x = x.view(-1, self.totalPooledSize)
        x = F.leaky_relu(self.encodeFC(x))
        mean = self.encodeMeanFC(x)
        varianceLog = self.encodeVarianceFC(x)
        return (mean, varianceLog)

    def sampleFromNormal(self, mean, varianceLog):
        # samples from N(mean, variance)
        # but keeps differentiabilty
        # using reparametrization trick
        if self.training:
            std = varianceLog.exp().pow(0.5)  # Standard deviation
            eps = self.normalDist.sample((mean.shape[0],)) # Sample from normal distribution(0, 1)
            sample = eps.mul(std).add_(mean) # Use a trick to make this N(mean, varianceLog.exp())
            return sample
        else:
            return mean
    
    def decode(self, z):
        z = F.leaky_relu(self.decodeFC1(z))
        z = F.leaky_relu(self.decodeFC2(z))
        z = z.view(-1, self.convFeatures[1], IMG_HEIGHT//4, IMG_WIDTH//4)
        z = self.decodeUnPool1(z, indices=self.indices2)
        z = F.leaky_relu(self.decodeUnConv1(z))
        z = self.decodeUnPool2(z, indices=self.indices1)
        z = F.leaky_relu(self.decodeUnConv2(z))
        z = F.leaky_relu(self.decodeFinalConv(z))
        return z

    def forward(self, x):
        # Find latent probablity for thix x, sample from its distribution and decode
        mean, varianceLog = self.encode(x)
        z = self.sampleFromNormal(mean, varianceLog)
        return (self.decode(z), mean, varianceLog)

    def get_loss(self, resultX, x, mean, varianceLog):
        # MSE and KL-divergence - see lecture
        MSEloss = F.mse_loss(resultX, x)
        KLloss = -0.5 * (varianceLog - mean.pow(2) - varianceLog.exp() + 1).sum()
        return MSEloss + KLloss

def train(model, epoch):
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        
        optimizer.zero_grad()
        (resultData, mean, logvar) = model(data)
        loss = model.get_loss(resultData, data, mean, logvar)
        loss.backward()
        train_loss += loss.data.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
        
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epochNum):
    model.eval()
    test_loss = 0

    for i, data in enumerate(test_loader):
        with torch.no_grad():
            (recon_batch, mu, logvar) = model(data)
            test_loss += model.get_loss(recon_batch, data, mu, logvar).data.item()

    test_loss /= len(test_loader.dataset)
    print('<> Test loss: {:.4f}'.format(test_loss))


def genComparison(model, epochNum):
    model.eval()
    print("Generating encode comparison...")

    examplesNum = 8

    for _, batchData in enumerate(test_loader):
        with torch.no_grad():
            dataToFeed = batchData
            (resultData, _, _) = model(dataToFeed)
            resultData = resultData.view(-1, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

            if not os.path.exists('results'):
                os.makedirs('results')

            save_image(torch.cat([dataToFeed[0:examplesNum], resultData[0:examplesNum]]),
            "results/encodeComparison" + str(epochNum) + ".png")
        break
        

def genSamples(model, epochNum):
    print("Generating samples...")
    model.eval()
    normalDist = torch.distributions.Normal(torch.zeros(LATENT_DIM), torch.ones(LATENT_DIM))

    with torch.no_grad():
        normalSamples = normalDist.sample((BATCH_SIZE,)) * 0.001
        # Turns out most of the samples are ~ 10^-3 
        # I dont know why will have to inspect
        samples = model.decode(normalSamples)
        samples = samples.view(-1, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

        if not os.path.exists('results'):
                os.makedirs('results')

        save_image(samples.data, "results/sample" + str(epochNum) + ".png")

(train_loader, test_loader) = get_FreyFace_data_loader(BATCH_SIZE)
#(train_loader, test_loader) = get_MNIST_data_loader(BATCH_SIZE)
model = VariationalAutoEncoder()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

for epochNum in range(TRAIN_EPOCHS):
    print("=================[ EPOCH #" + str(epochNum) + " ]=================")
    train(model, epochNum)
    test(epochNum)

    if epochNum%10 == 0:
        genComparison(model, epochNum)
        genSamples(model, epochNum)
