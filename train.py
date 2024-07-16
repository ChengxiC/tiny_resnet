# Chengxi Chu, Universiti Malaya
import torch
from torch import nn, optim
from matplotlib import pyplot as plt


def train(model, device, lr, epochs, plot, loader_train):

    model = model.to(device)
    learning_rate = lr
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epoch_list = []
    loss_list = []

    # start training
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (x, labels) in enumerate(loader_train):
            x = x.to(device)
            labels = labels.to(device)
            logits = model(x)
            loss = criterion(logits, labels)
            epoch_loss += loss.item()

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / (batch_idx+1)

        epoch_list.append(epoch+1)  # for plot
        loss_list.append(avg_loss)  # for plot
        print(f'epoch:{epoch+1}; loss:{avg_loss}')

    if plot is True:
        plt.plot(epoch_list, loss_list, linestyle='--')
        plt.title('epoch vs loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

    torch.save(model, './pretrained_model/model.pth')

















