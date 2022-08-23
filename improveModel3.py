import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch import nn
from helper_functions import *
import requests


def model_3_imporved():
    # Make device agnositic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device runing is: {device} * 3")

    print(torch.__version__, ' Torch version ')

    print(
        f"What patterns could you draw if you were given an infinite amount of a straight line and non-straight linge "
        f"Or in ML terms, an infinite (but really it is finite) o f linear and non-linear functions")

    # 6.1 Recreating non-linear data (red an blue cercles)

    # Make data

    n_samples = 10000
    X, y = make_circles(n_samples,
                        noise=.03,
                        random_state=42)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/output/makeData_nonlinearity.png")

    # Convert data to tensors and split

    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)

    # split the datasets
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=.2,
                                                        random_state=42)
    print(X_train[:4], y_train[:4])
    print(len(X_train), len(X_test), len(y_train), len(y_test))

    print("test " * 4)

    print(
        f" Non leanear a graph with non straight line >> Linear >>> Non Linear : Linear data , non linear data make_circles"
        f"ML can work with 100 of dimension ")

    class CircleModelV3(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=2, out_features=10)
            self.layer_2 = nn.Linear(in_features=10, out_features=10)
            self.layer_3 = nn.Linear(in_features=10, out_features=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            # Where should we put our non-linear Activation function ?
            return self.layer_3(self.layer_2(self.relu(self.layer_1(x))))

    model_3 = CircleModelV3().to(device)
    print(model_3, '\n \n ', next(model_3.parameters()))

    print(model_3.state_dict())
    print(f"Writing a training and test loop ... "
          f"Artitial Neural Network >>> Binaray classification problem: "
          f"email spam/not"
          f"credict cares fraud / not "
          f"insurance claim: at fault, not  ")

    # Set up loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model_3.parameters(),
                                lr=0.1)

    # Train a model with non linearity
    # Random Seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Put all the data on the target device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Loop through data
    epochs = 1000

    for epoch in range(epochs):
        # Training
        model_3.train()

        # 1. Forward pass
        y_logits = model_3(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))  # logits -> prediction probabilities -> prediction labels

        # Calculat the Loss
        loss = loss_fn(y_logits, y_train)  # BECWithLogitsLoss (takes in logits as first input )
        acc = accuracy_fn(y_true=y_train,
                          y_pred=y_pred)

        # optimizer zero_grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Step teh optimizer
        optimizer.step()

        ## Testing
        model_3.eval()
        with torch.inference_mode():
            test_logits = model_3(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))

            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test,
                                   y_pred=test_pred)

        # Print what's happening
        if epoch % 100 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss:.5f} | Train_acc: {acc:.2f}% |  Test Loss: {test_loss:.5f} | Test_acc: {test_acc:.2f}% ")

    # 6.4 Evaluate the model >>> trained with non-linear activations functions
    print("Makes predictions " * 2)
    model_3.eval()
    with torch.inference_mode():
        y_test_pred = torch.round(torch.sigmoid(model_3(X_test))).squeeze()

    print(y_test_pred[:10], y_test[:10])
    print(y_test_pred[:-10] == y_test_pred[:-10])

    # Plot decision bounaries
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model_3.cpu(), X_train.cpu(), y_train.cpu())  # Model 1 has no non-linearity
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model_3.cpu(), X_test.cpu(), y_test.cpu())  # Model 3 has non-linearity
    plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/output/ourFirst_non_linearModel_imporved.png")