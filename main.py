'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
from utils import progress_bar, get_lr, recycle
from efficientnet_pytorch import EfficientNet

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--gamma', default=0.1, type=float, help='lr decay')
parser.add_argument('--wd', default=1e-6, type=float, help='weights decay')
parser.add_argument('--ne', default=30, type=int, help='number of epochs')
parser.add_argument('--nsc', default=10, type=int, help='number of step for a lr')
parser.add_argument('--batch_split', default=1, type=int, help='spliting factor for the batch')
parser.add_argument('--batch', default=32, type=int, help='size of the batch')
parser.add_argument('--alpha', default=0.1, type=float,
                    help='mixup interpolation coefficient (default: 1)')

args = parser.parse_args()

#To get reproductible experiment
torch.manual_seed(0)
np.random.seed(0)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device : ",device)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=160,scale=(0.6,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


transform_test = transforms.Compose([
    transforms.Resize(200),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
micro_batch_size = args.batch // args.batch_split


criterion = nn.CrossEntropyLoss()

def mixup_data(x, y, alpha=1.0,lam=1.0,count=0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if count == 0:
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Training
def train(epoch,trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    count = 0
    lam = 1.0
    optimizer.zero_grad()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if count == args.batch_split:
            optimizer.step()
            optimizer.zero_grad()
            count = 0
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,args.alpha,lam,count)
        outputs = net(inputs)
        loss =  mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss = loss / args.batch_split
        loss.backward()
        count +=1
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(testloader,namesave):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, namesave)
        

testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=1)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=micro_batch_size, shuffle=True, num_workers=1)        

net = EfficientNet.from_pretrained('efficientnet-b4', num_classes=10)
net = net.to(device)
namesave='./checkpoint/ckpt'
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
lr_sc = lr_scheduler.StepLR(optimizer, step_size=args.nsc, gamma=args.gamma)
for epoch in range(0, args.ne):
    train(epoch,trainloader)
    lr_sc.step()
    
print("Test accuracy : ")
test(testloader,namesave)
    
    


