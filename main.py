from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import util
import model

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='Music')

# Data
parser.add_argument('--audio_length', type=int, default=133623)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--resume', '-r', action='store_true')
# Training
parser.add_argument('--learning_rate', type=int, default=0.1)
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--log_frequency', type=int, default=25)
parser.add_argument('--save_frequency', type=int, default=20)

config = parser.parse_args()
use_cuda = torch.cuda.is_available()

if config.resume:
    print('-- Resuming From Checkpoint')
    assert os.path.isdir('checkpoint'), '-- Error: No checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/music.nn')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    net = model.net(config)
    start_epoch = 1

if use_cuda:
    net = net.cuda()
    cudnn.benchmark = True

util.print_network(net)
net.train()
best_acc = 0
criterion = torch.nn.CrossEntropyLoss()
Optimizer = torch.optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9)

trainloader = util.get_loader(config, './train')
testloader = util.get_loader(config, './test')

def train(epoch):
    print('-- Current Training Epoch %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        Optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        Optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        util.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.2f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    print('-- Current Testing Epoch %d' %(epoch))
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        util.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.2f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('-- Got Best Result. Saving Model. Test Accuracy: %.2f%%' %(acc))
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/pytorch.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)