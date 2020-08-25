'''Aggregate Trained models on CIFAR10 with PyTorch.'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
from utils import progress_bar
from collections import Counter
from efficientnet_pytorch import EfficientNet

parser = argparse.ArgumentParser(description='PyTorch CIFAR 10 aggregation')
parser.add_argument('--ns', default=10, type=int, help='number of samples')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device : ",device)

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.Resize(200),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Model
print('==> Building model..')
net = EfficientNet.from_pretrained('efficientnet-b4', num_classes=10)
net = net.to(device)
batch_size=10
nclasses=10
nsplit=args.ns

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

namesave='./checkpoint/ckpt'
def extractoutputs(loader,namesave='./checkpoint/ckpt',batch_size=1,nsplit=10,nclasses=10,nbobs=10000):
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    outputsnet = torch.zeros(nsplit,nbobs,nclasses)
    predictsnet = torch.zeros(nsplit,nbobs,dtype=torch.int)
    for i in range(0,nsplit):
        print('split ',i)
        correct = 0
        namesaveb=namesave+str(i)+'.t7'
        checkpoint = torch.load(namesaveb)
        net.load_state_dict(checkpoint['net'])
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                output = F.softmax(net(inputs),dim=1)
                _, predicted = output.max(1)
                correct += predicted.eq(targets).sum().item()
                output = output.to('cpu')
                indice=batch_idx*batch_size                
                outputsnet[i,indice:(indice+output.size(0))]=output
                predictsnet[i,indice:(indice+output.size(0))]=predicted
        print("Test accuracy : ", 100.0*correct/nbobs)
    return outputsnet, predictsnet
    


def find_majority(predictsnet):
    majvote=torch.zeros(predictsnet.size(1),dtype=torch.int)
    for i in range(predictsnet.size(1)):
        votes = Counter(predictsnet[:,i].tolist())
        majvote[i]=votes.most_common(1)[0][0]
    return majvote



outputsnet, predictsnet = extractoutputs(testloader,namesave,batch_size,nsplit,nclasses)
moytest = torch.mean(outputsnet,dim=0)
moyharmotest = torch.mean(outputsnet.log(),dim=0)
majtest = find_majority(predictsnet)



def testaggregatemoy(testloader,moytensor):
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            _, predicted = moytensor[batch_idx:(batch_idx+1)].max(1)
            correct += predicted.eq(targets).sum().item()
    return correct/moytensor.size(0)

def testaggregatemaj(testloader,majtensor):
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            correct += majtensor[batch_idx].eq(targets).sum().item()
        return correct/majtensor.size(0)
            
                

    
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
erragregmoy = testaggregatemoy(testloader,moytest)
erragregmoyharmo = testaggregatemoy(testloader,moyharmotest)
erragregmaj = testaggregatemaj(testloader,majtest)

print('==> Error of soft aggregation: ',erragregmoy)
print('==> Error of soft harmonic aggregation: ',erragregmoyharmo)
print('==> Error of hard aggregation: : ',erragregmaj)



