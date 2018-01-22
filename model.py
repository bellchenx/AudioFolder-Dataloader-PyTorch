'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class net(nn.Module):
    def __init__(self, config):
        super(net, self).__init__()
        self.length = config.audio_length
        self.conv1 = nn.Conv1d(2, 200, 1000, 500)
        self.pool1 = nn.MaxPool1d(10, 10)
        self.fc1   = nn.Linear(160, 10)
        self.fc2   = nn.Linear(10, 2)

    def step(self, input, hidden=None):
        input = self.inp(input.view(1, -1)).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

    def forward(self, inputs, hidden=None, force=True):
        out = self.pool1(F.relu(self.conv1(x)))
        out = out.view(out.size(0), -1)
        
        if force or steps == 0: steps = len(out)
        outputs = Variable(torch.zeros(steps, 1, 1))
        for i in range(steps):
            if force or i == 0:
                input = inputs[i]
            else:
                input = output
            output, hidden = self.step(input, hidden)
            outputs[i] = output
        return outputs, hidden

        
        out = F.sigmoid(self.fc1(out))
        out = self.fc2(out)
        return out