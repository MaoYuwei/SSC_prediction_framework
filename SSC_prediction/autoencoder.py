import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        #Encoder
        self.enc1 = nn.Linear(in_features=18, out_features=64)
        self.enc2 = nn.Linear(in_features=64, out_features=4)

        #Decoder
        self.dec1 = nn.Linear(in_features=4, out_features=64)
        self.dec2 = nn.Linear(in_features=64, out_features=18)

    def forward(self, x):
        encoder = F.relu(self.enc1(x))
        encoder = F.relu(self.enc2(encoder))
        decoder = F.relu(self.dec1(encoder))
        decoder = F.relu(self.dec2(decoder))

        return encoder, decoder

def training(model, trainset, Epochs):
    for epoch in range(Epochs):
        _, outputs = model(trainset)
        loss = criterion(outputs, trainset)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch {} of {}, Train Loss: {}'.format(
            epoch+1, Epochs, loss))

if __name__ == '__main__':
    Epochs = 500
    Lr_Rate = 1e-3

    path = '../../auto/02/ret/'
    train_features = []

    first_list = list(range(0, 91))
    second_list = list(range(0, 91))
    for first in first_list:
        for second in second_list:
            Fp1 = np.loadtxt(path + str(first)+'-'+str(second)+'-0' + '_Fp_tau.txt')[0]
            F1 = np.loadtxt(path + str(first)+'-'+str(second)+'-0' + '_F_tau.txt')[0]

            tmp = np.array([first, second])

            train_features.append(np.concatenate([Fp1, F1]))

    train_features = np.array(train_features)
    train_features = torch.FloatTensor(train_features)

    model = Autoencoder()
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Lr_Rate)
    training(model=model, trainset=train_features, Epochs=Epochs)

    w = list(model.parameters())
    print(len(w))
    weights = {}
    for i in range(8):
        weights[str(i)] = w[i].detach().numpy().tolist()
        print(w[i].detach().numpy().shape)

    with open('weights.json', 'w') as f:
        json.dump(weights, f)




