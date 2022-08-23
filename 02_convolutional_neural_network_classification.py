print

print(f"Classification>> Spam/ not spam... binary classifier problem .. \n "
      f"Multiclass classifier problem: foods classification >>> 10 different foods >>> \n "
      f"ImageNet a 1000 object classes : multiclass \n "
      f"Multilabel classification: 3 labels, 2 labels, categoreies >> most relevant categories >> \n "
      f"")

print(f"Architecture of a neural network classsification model \n "
      f"Input shapes and output shapes of a classification model (features and labels) \n "
      f"Creating custom data to view, fit on and predct on \n "
      f"Stpes in modeling \n "
      f"Creating a model, setting a loss function and optimiser, creating a training loop, evaluating a model \n "
      f"Saving and loading models \n "
      f"Harnessing the power of non-linearity \n "
      f"Different classification evalualtion methods .. "
      f"Numerical encoding >>>> inputs >> ML Algorithm >>> outputs... predicted probabilities .. "
      f""
      f""
      f"image widht>224, height>224, c=3 >>> input (ML Algorithm) output >> (predictions probabilities) ... "
      f"numerical encoding >>> output numerical encoding >>> "
      f"batch_size, color_channels, widht, height"
      f"shape = [None, 3, 224, 224] "
      f"shape = [ 32, 3, 224, 224] "
      f"32 is a very common batch size >>> "
      f"output shape is 3 [0.97, .0, .3] "
      f""
      f"High level Architecture of neural network >>> classification model "
      f""
      f"\n "
      f"\n"
      f"input layer shape(in_features) >> num of features (age, sex, height, weight, smoking status) "
      f"\n"
      f"hidden layers nn.Linear(in, out)"
      f"neuron per hidden layers (10 to 555)"
      f""
      f"output layer shape (out_features) "
      f"hidden layer activations "
      f"Non-linear Activations weight, biases"
      f"output activation >> sofmax for multiclass classification"
      f"loss function : measures how wron gour model predictions to idial targets ... binar cross entropy, "
      f"optimizer SGD, Adam , torch.optim  "
      f"\n")

print(f"NN classification: "
      f"classification is a problem of predicting whether somethion is one thin or another >> multiple options "
      f"1. Make classification data and get it ready ... ")


import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch import nn
import requests
 #Make device agnositic code
device = "cuda" if torch.cuda.is_available()  else "cpu"
print(f"Device runing is: {device} * 3")

print(torch.__version__, ' Torch version ')






#Make 100 samples

n_samples = 1000

#create circles
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)
print(len(X), len(y))
print(f"First 5 samples of X: {X[:5]} featues \n and {y[:5]} labels \n")
print(y[:10])

#Make Dataframe of circle data
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})
print(circles.head())
print("Visualize "* 3)

#create a dirctors of
PATH_OUTPUT = Path("output")
PATH_OUTPUT.mkdir(parents=True, exist_ok=True)
PLOT_NAME = 'visualized_data.png'
SAVED_VISUALIZATION = PATH_OUTPUT / PLOT_NAME

plt.scatter(x=X[:,0],
            y=X[:,1],
            c=y,
            cmap=plt.cm.RdYlBu);
plt.savefig(SAVED_VISUALIZATION)



print("Data are too small enough to experiment on but ideal to learn a model called Toy datasets.... ")
print('Turning our data into tensors and split them on training and test sets')
###1.1 Check input and output shapes >>>>>> acquinted
print(X.shape, ' two features', y.shape, ' scalar y ')
print("View the first example of features and labels .."
      )
X_samples = X[0]
y_sample = y[0]

print(f" Values for one sample of X: {X_samples} and same for y: {y_sample}")
print(f" shape  for one sample of X: {X_samples.shape} and same for y: {y_sample.shape}")


#1.2 Turn data into tensors and create train test splits
print(type(X), X.dtype , '  since Pytorch default type is float32 so we need to change all tensors to float32 instead of '
                         'float64')
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
print(X[:5], y[:5])
print(type(X), X.dtype, type(y), y.dtype)

#3. Split data into train and test split uign random_state>>> torch.random_seed(42) == random_state=42
X_train,X_test,  y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, # 20% test, 80% train
                                                     random_state=42)

print(len(X_train), len(X_test), len(y_train), len(y_test))

#6.5 Laying out steps for modeling and settings up device agnositic code
print(f"Training set >> let's model the patters.... test dataset > evaluating model's patterns.. ")
#2. Building a model to classsify a red and blue dots...
print(f"1. Setup device agnositic code so our code wil run on an accelerator (GPU) if there is one "
      f"2. Construct a model (by subclassing nn.Module) "
      f"Define a loss function and optimizer "
      f"Create a training and test loop")
# 1. Sub class nn.Module (almost all module on PyTorch used nn.Module)
#2. create 2 'nn.Linear()' layers that are capable of handling the shapes of our data
#3. Defines a 'forward()' method that outlines the forward pass (or forward computation) of the model
#4. Instantiate an instance of our model class and send it to the target device

print("1. Construct a model that subclass nn.Module")
print("2. inside the constructor, create 2 nn.Linear() layers that are capable of handling the shapes")

class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # in 2 features >> (hidden units) upscale to 5 features >>
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # Takes in 5 features>> ouputs 1 features (same shpae as y)

        # self.two_linear_layers = nn.Sequential(
        #     nn.Linear(in_features=2, out_features=5),
        #     nn.Linear(in_features=5, out_features=1)
        # )
    def forward(self, x):
        return self.layer_2(self.layer_1(x)) # X -> layer_1 -> layer_2 -> output layer
        # return two_linear_layers(x)

print('Instatiate a model class and send to a device0.')
model_0 = CircleModelV1().to(device)
print(model_0)
print(next(model_0.parameters()))

print("Let's replicate the model above using nn.Sequential")
print(model_0.state_dict(), '\n')


#Make predictions >>>
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape} \n "
      f"Length of test samples: {len(X_test)}, Shape: {X_test.shape} \n "
      f"\n First 5 predictions: \n {untrained_preds[:5]} \n "
      f"\n First 4 labels: \n {y_test[:5]} ")



print(f"Lossfunction: and ooptimizer is a problem specific "
      f"for regression: MSE, MAE, "
      f"for classification: BCE, CCE \n "
      f"The logits layer means the layer that feeds in to sofmax or(other such normalization). THe output of the softmax are the probabilities"
      f"for the classification task and its input is logits layer ... \n "
      f"the logits layer typically produces values from - infinity to + infinity "
      f"and the softmax layer transforms it to values from 0 to 1 \n "
      f"optimizer SGD or Adam ... ")

#torch.nn.BECWithLogitsLoss() >>>> logits >>>> troch.optim >>> differnt ooptimizers

#Set up loss function
#loss_fn = nn.BCELoss() # THis requires input to have gone through the sigmoid activation function prior to the BCELoss
# nn.Sequential(
#     nn.Sigmoid(),
#     nn.BCELoss()
# )
loss_fn = nn.BCEWithLogitsLoss() # has sigmoid activation function >>> 1/1+e^-x
print("BCEWithLogitsLoss: combines a Sigmoid layer and the BCELoss in one single class.. \n "
      "nn.Sequential("
      "nn.Sigmoid(),"
      "nn.BCELoss()")
print('For multiclass classification: softmax. >>>> for binary classification: sigmoid activation')

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr  = 0.1)

print(model_0.state_dict())

#Calculate the accuracy
# out of 100 exampes are correct

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
print("70. Going from model logits to prediction probabilities to prediction labels ... ")

#Training a model --- training and test loop
print(f"1. Forward passs "
      f"Calculate the loss "
      f"Optimizer Zero grad "
      f"Loss backward (backpropogation)"
      f"Optimizer step (Gradient descent) ")

#3. 1 Going from raw logits >>> prediction probabilities >> prediction labels
print(f"Our model outputs are going to be logits  .. "
      f"convert those logits to prediction probabilities by passing them to "
      f"some kind of activations function e.g. Sigmoid for binary class classification and "
      f"softmax for multiclass classifiction ... "
      f"Convert our model's prediction probabilities to prediction labels "
      f"by either rounding them for binary classification or taking the 'argmax()' the output of the softmax  ")

#View the first 5 outputs of the forward pass onthe test dataset
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
print(y_logits)
print(y_test[:5])

#Use sigmoid activations on our model to trun them into prediction probabilities
y_pred_probs = torch.sigmoid(y_logits)
print(y_pred_probs)

print(torch.round(y_pred_probs))
print(f"For our prediction probability values, we need to perform a range-style rounding on them "
      f"y_pred_probs>=0.5, y =1 (class: 1)"
      f"else: class: 0")

#Find the predicted labels
y_preds = torch.round(y_pred_probs)

#in full (logits -> pred probs -> pred labels)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

#Check for equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

#Get rid of extra dimension
print(y_preds.squeeze())

print(y_test[:5])

#Building a training and test loop
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#set the numper of epochs
epochs = 100

#Put data to the target device
X_train, y_train = X_train.to(device) , y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


#Build our training and evalualtion loop

##Training
for epoch in range(epochs):
    #Training
    model_0.train()

    #1. Forward pass
    y_logits = model_0(X_train).squeeze() # logits
    y_pred = torch.round(torch.sigmoid(y_logits)) # convert logits into prediction probabilities > pred y_pred_labels


    #2. Calculate the loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), #nn.BCELoss expects prediction probabilities as input
    #                y_train)
    loss = loss_fn(y_logits, #nn.BCEWithLogitsLoss expects raw logits as input
                   y_train)

    acc = accuracy_fn(y_true=y_train,
                      y_pred = y_pred)

    #3. Optimizer zero grad
    optimizer.zero_grad()

    #4. Loss backward (backpropogation)
    loss.backward()

    #5. Optiimezer steps
    optimizer.step()

    ### Testing
    model_0.eval()

    with torch.inference_mode():
        #1. Forward passs
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        #2. Calculate the test loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)

        test_accuracy = accuracy_fn(y_true= y_test,
                                    y_pred= test_pred)


    #Print what's happening
    if epoch % 10 ==0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Train_acc: {acc:.2f}% |  Test Loss: {test_loss:.5f} | Train_acc: {test_accuracy:.2f}% ")





print(circles.label.value_counts())

#4. Visualize -> Make predictions and evaluation
print(f"evaluation ... Downloading helper functions from lear PyTroch repos  > ")

if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downolaoding helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_function.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary


#Plot descesion boundry of the model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/output/MakeCircles_Model_0.png")

print(f"Improving a model from (model's prespective) to (model)"
      f"Add more layer >>> to learn more patters  "
      f"Add more hidden units - from 5 to 10 hiddens "
      f"Fit for longer ... "
      f"changing the activation function "
      f"Change the learning rate ... "
      f"Change the loss function ... ")

#5. Creating a new model with imporving >>> these options... are all from model prespective .. as the deal
#with model >>> not data... Because these options are values we
# ML engineers and data scienteits can be hyperparamters ...
#1. adding more hidden units - from 5 to 10 hiddens
#2. INcrease the number of layers from 2 -> 3
#3. Increase the number of epochs: 100 -> 1000
# experiment tracking


class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        # z = self.layer_1(x)
        # z = self.layer_2(z)
        # z = self.layer_3(z)
        return self.layer_3(self.layer_2(self.layer_1(x)))


model_2 = CircleModelV2().to(device)
print(model_2, ' \n \n mofrl ')
print(model_2.state_dict())
#Creaet a loss function
loss_fn = nn.BCEWithLogitsLoss()
#Create optimizer
optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr = 0.1)
#Train for longer
epochs = 1000

# Put data on the target Device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Training Testing loop

for epoch in range(epochs):
    ### Training
    model_2.train()

    #1. Forward pass
    y_logits = model_2(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> pred probabilities -> prediction labels

    #2. Claculate the loss/acc
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true= y_train,
                      y_pred = y_pred)

    #3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loass backward (backpropogation)
    loss.backward()

    # 5. Optimizer step (gradient descent)
    optimizer.step()

    ## Testing
    model_2.eval()
    with torch.inference_mode():
        #1. Forward pass
        test_logits = model_2(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits)) # sigmoid activation -> to convert binary Classification To predict probabilities
        # Calculate the loss
        test_loss = loss_fn(test_logits,
                            y_test)

        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

        #Pring out what's happenin
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.2f}, Test accuracy: {test_accuracy:.2f}%")


#Plot descesion boundry of the model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_2, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_2, X_test, y_test)
plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/output/MakeCircles_Model_2.png")


print(f"Creating a straight line dataset to see if the model learning ... ")
## 5.1 Preparing data to see if our model can fit a straight line
# Troubleshoot a larger problem is to test out a smaller problem
print("Wy stop runing " * 3)
#Create some data
weight = 0.7
bias = .3
start = 0
end = 1
step = .01

#create data
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias # Linear regression formula (without epsilon)

#check data
print(len(X_regression))

print(X_regression[:5], y_regression[:5])

#Create Train and test splits
train_split = int(.8 * len(X_regression))
X_train_regression, y_train_regression = X_regression[:train_split], X_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], X_regression[train_split:]

#Check the length
print(X_train_regression.shape, y_train_regression.shape, X_test_regression.shape, y_test_regression.shape)
print(len(X_train_regression), len(X_train_regression), len(y_train_regression))

plot_predictions(train_data=X_train_regression,
                 train_labels= y_train_regression,
                 test_data=X_test_regression,
                 test_labels=y_test_regression)

#Plot descesion boundry of the model

 # 5.2 Adjust the mdoel_2 to fit the straight line data
print(X_train_regression[:10]), print(y_train_regression[:10])

#Same architecture as model_2 (but using nn.Sequential())

model_2 = nn.Sequential(
 nn.Linear(in_features=1, out_features=10),
 nn.Linear(in_features=10, out_features=10),
 nn.Linear(in_features=10, out_features=1)
).to(device)
print(model_2)
#loss and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.01)

#Train the model
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#set the number of epoch
epochs = 1000

#put the data in the target device
X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

# Training
for epoch in range(epochs):
    y_pred = model_2(X_train_regression)
    loss = loss_fn(y_pred, y_train_regression)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Testing
    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)
        test_loss = loss_fn(test_pred, y_test_regression)

    # Pring out what's happenin
    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f} ")

print("Evaluate the model >>>>  Turn on evaluation mode ")

model_2.eval()
#Make prediction inference
with torch.inference_mode():
    test_pred = model_2(X_test_regression)

#plot data and prediction
plot_predictions(train_data= X_train_regression.cpu(),
                     train_labels= y_train_regression.cpu(),
                     test_data= X_test_regression.cpu(),
                     test_labels=y_test_regression.cpu(),
                 predictions=test_pred.cpu(),
                 PATH="/home/mhamdan/DeepLearning_PyTorch_2022/output/linearModel_visualization.png")

print(f"Non linearity: \n " * 4)

#The missing piece : non-linearity

print(f"What patterns could you draw if you were given an infinite amount of a straight line and non-straight linge "
      f"Or in ML terms, an infinite (but really it is finite) o f linear and non-linear functions")


#6.1 Recreating non-linear data (red an blue cercles)

#Make data

n_samples = 1000
X, y = make_circles(n_samples,
                    noise=.03,
                    random_state=42)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/output/makeData_nonlinearity.png")

#Convert data to tensors and split

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
#Building a model with non linearity
print(f" Non leanear a graph with non straight line >> Linear >>> Non Linear : Linear data , non linear data make_circles"
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

#Set up loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_3.parameters(),
                            lr = 0.1)

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
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels


    # Calculat the Loss
    loss = loss_fn(y_logits, y_train) # BECWithLogitsLoss (takes in logits as first input )
    acc = accuracy_fn(y_true= y_train,
                      y_pred= y_pred)

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
        test_acc = accuracy_fn(y_true= y_test,
                               y_pred = test_pred)

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
plot_decision_boundary(model_0.cpu(), X_train.cpu(), y_train.cpu()) # Model 1 has no non-linearity
plt.subplot(1, 2,2)
plt.title("Test")
plot_decision_boundary(model_3.cpu(), X_test.cpu(), y_test.cpu()) # Model 3 has non-linearity
plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/output/ourFirst_non_linearModel.png")

print("Model_3 imporved .... " * 5)

# # from improveModel3 import model_3_imporved
#
# # model_3_imporved()


print(f"Replicationg non-linear activation functions : "
      f"Neural network, rather than us telling the model what to learn, we give it the tools to discover "
      f"patterns in dta and patterns on its own... "
      f"And these tools are linear and non linear functions \n \n ")

# Create a tensor >>>
A_int = torch.arange(-10, 10, 1)
print(A_int.dtype, A_int.type, A_int.shape, len(A_int), A_int[:4])

A = torch.arange(-10, 10, 1.0)
print(A.dtype, A.type, A.shape, len(A), A[:4])

A = torch.arange(-10, 10, 1, dtype=torch.float32) # Since PyTorch's default data type is int64
print(A.dtype, A.type, A.shape, len(A), A[:4])

# Visualize the tensor
plt.plot(A)
plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/output/linearTensor.png")

plt.plot(torch.relu(A))
plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/output/non-linearTensor.png")

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.tensor(0), x) # inputs must be tensors
plt.plot(relu(A))
plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/output/non-linearTensor_replicate_relu.png")

print("Sigmoid >>> 1 / 1 + exp(-x) ")
def segmoid(x):
    return 1 / (1 + torch.exp(-x))

plt.plot(torch.sigmoid(A))
plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/output/non-linearTensor_Sigmoid_called.png")

plt.plot(relu(A))
plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/output/non-linearTensor_replicate_sigmoid.png")

print('Putting all togather >>> ... Built a multi-class classification ...')