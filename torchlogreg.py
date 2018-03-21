import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data

num_classes = 2
batch_size = 100
learning_rate = 0.001
input_size = 100

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(6, 100)
    
    def forward(self, x):
        return F.sigmoid(self.linear(x))


class GameDataSet(Dataset):

    def __init__(self, csv_path):
        # Transforms
        self.to_tensor = transforms.ToTensor()
        self.dataset = pd.read_csv(csv_path, encoding = "utf8")
        self.dataset.dropna(how='all', inplace = True)

        #drop columns with strings
        cols = [2,3,4]
        self.dataset.drop(columns = ['name','developer','publisher'], axis=1,inplace=True)
        self.data_arr = np.asarray(self.dataset.iloc[:,[0,1,2,3,4,5]])

        self.label_arr = np.asarray(self.dataset.iloc[:,6])

        self.data_len = len(self.dataset.index)

    def __getitem__(self, index):

        data = self.data_arr[index]

        # Transform image to tensor
        data_as_tensor = torch.FloatTensor(data)


        #get label of row
        label = self.label_arr[index]

        return (data_as_tensor, label)

    def __len__(self):
        return self.data_len



game_dataset = GameDataSet('dataUpdated.csv')

game_dataset_loader = torch.utils.data.DataLoader(dataset=game_dataset,
                                                    batch_size=100,
                                                    shuffle=True)

model = LogisticRegression(input_size, num_classes)
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

#Train model
for data, label in game_dataset_loader:
    
    data, label = Variable(data), Variable(label)
    # Forward + Backward + Optimize
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, label)
    loss.backward()
    optimizer.step() 
