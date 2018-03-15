import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

num_classes = 2
batch_size = 100
learning_rate = 0.001

X = pd.read_csv('dataUpdated.csv', encoding = "utf8")





#print(X.shape)
X.dropna(how='all', inplace = True)

# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        out = self.linear(x)
        return out

model = LogisticRegression(input_size, num_classes)


