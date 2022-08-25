#Object detection where is the thing I am loooking for ... hit and run ... 
# What are the different sections in this image >>> Segmentation (semantic mask)
# Tesla > Apple >> Tesla 8 Cameras >>> 3 Dimenstional (Vector space) long strong of numbers .. Vector space >>> Where to drive next >>> ML is probabilistics 
#>>> -- What CV to wornk with >>> type of model >>> CNN . Transformer vision > Deep learning model 
# Input . output shapes >>> shape  ( batch_size, width, height, color_channels] shape=[Non, 224, 224, 3] , shape [32, 224, 224, 3]
# encode data >> feed it to a CNN model >>> ouput shape... 
# Grayscale image >> ML algor >> output >> [0, 0, 8, 0, ...] ...> [batch_size, colour_channels, height, width] PyTorch default to reperesent colour channel first >>> however >> some other represent data in color channel last. data are the same.. 

# torch.transforms.  torch.nn.Module torch.utils.tensorboard . torch.save . load

# What is a covolutional neural network (CNN) >>> hyperparamters: input layer>>> convlutional layer (layers math opreations window optersation ) nn.Conve2d> out = bias(cou) + sum (weight * input : B + sum(W * X)  operations layer by layer into some useLLL
# hiddent activation / non-linear / >>> pooling layer .. ?  output layer / linear layer : >>> 
# forward metho return x ... code ....

# import dependencies Computer vision libraries
# torchvision >> package consist of popular dataset model ... augmenting images / predtrained weights, datasets, utils, operators
#torchvision.datasets - get datasets and dataloading function for computer vision ... tor
# torchvision.model - get pretained weights that you levarage for another problem
# torchvision.transforms - functions for manipulating your vision data (iamges) to be suitable for use with an ML model
# torch.utils.data.Datasets - Base datasets class for PyTorch
#torch.utils.data.DataLoader - creates a pPython iterabel oever datasts
import torch
from torch import nn
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor # Convert a PIL image or numpy.ndarray to tensor
from pathlib import Path
from helper_functions import *
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm import tqdm

#Check versions
print(torch.__version__)
print(torchvision.__version__)


# getting datasts >> check input and outputs rec
#. 1. getting a dataset : FashionMNIST) ...hello world MNIST which used for trying to find out to use CN to figure out post codes
# grayscale images of clothes ... dimensitonaliy group >> intractive >> MNIST overused >> FashionMINEST .. Buuilt-in datasets To download a dataset ... imagenet >> gold standed >> practice on .. built Module ... torchvision.datasets.FashinMNIST (yes, no)
#Transform the data >>> to tensors. .. target transform >>> .. custom datasets ... how you get your data in the right format ..
# Fashion MNIST 

DATASET_PATH = Path("datasets")
DATASET_PATH.mkdir(parents=True, exist_ok=True)
DATASET_NAME = "FashionMNIST"
SAVED_DATASET = DATASET_PATH / DATASET_NAME

# Setup training data
train_data = datasets.FashionMNIST(
    root = SAVED_DATASET, # where to download data to?
    train = True, # do we whant the training dataset 
    download=True, # do we need it
    transform=ToTensor(), # how do we want to transform the data
    target_transform=None # how do we want to transform the labels/targets 
)

test_data = datasets.FashionMNIST(
    root= SAVED_DATASET,
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)


print(len(train_data), len(test_data))

# see the first training examples of ToTensor >>>> convert a PIL in numpy array to Tensor >> of shape (C H W) in range [0.0, 1.0] 
image, label = train_data[1]
print(image, '\n', label)


class_names = train_data.classes
print(class_names)

class_to_idx = train_data.class_to_idx
print(class_to_idx)
targetss = train_data.targets
print(targetss)
    
#shape 
print(f"Image shape: {image.shape}  -> [color_channels, height, width], \n label is {targetss}") 



#Visualize a random dataset -. decifier ... 

# Input and output shapes >> ToTensor > input shape [ None, 1, 28, 28] [NCHW] > CNN > output [ 10 shape ] ... 
DATASET_PATH = Path("visualization")
DATASET_PATH.mkdir(parents=True, exist_ok=True)
DATASET_NAME = "img1.png"
SAVED_DATASET = DATASET_PATH / DATASET_NAME
# 1.2 Visualize data
image, label = train_data[1]
print(f"Image shape: {image.shape}")
plt.imshow(image.squeeze())#data format color channel format >> hight, width... or color channel is last
plt.savefig(SAVED_DATASET)

DATASET_NAME = "img2.png"
SAVED_DATASET = DATASET_PATH / DATASET_NAME
plt.imshow(image.squeeze(), cmap='gray')
plt.title(class_names[label])
plt.axis(False)
plt.savefig(SAVED_DATASET)



DATASET_NAME = "randomimages.png"
SAVED_DATASET = DATASET_PATH / DATASET_NAME
# look some random images from dataset 
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9) )
rows, cols = 4, 4
for i in range(1, rows*cols+1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    #print(random_idx)
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)
plt.savefig(SAVED_DATASET)

print(f"98 DataLoader overview and understanding minibatches "
      f"Prepare DataLoader, "
      f"ourdata is in the form of PyTorch Datset {train_data} and \n test data {test_data} "
      f"the DataLoader return the datasets to  a pPython iterabel "
      f"More sepecifically, turn our data into batches (mini-batches) "
      f"1. It is more compuationally efficient, as in, your computing hardware may not be "
      f"able to look (store in memory) at 60000 images in one hit. So we break it down to 32 images at a time "
      f"(batch size of 32) "
      f"2. It gives our neural network more chances to update its gradients per epoch (get one update per epoch, neural network updates each 32 images instead of 60000 images ) "
      f""
      f"numper_samples/batch_size = number of batches"
      f"batchfy ")

# Preparing a DataLoader
# Setup the batch size hyperparamter

BATCH_SIZE = 32

#Turn dataset into iterabel
train_dataloader = DataLoader(dataset = train_data,
                              batch_size= BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(dataset = test_data,
                             batch_size= BATCH_SIZE,
                             shuffle=False)
print(f"It is important to test the model with ordered data (not shuffled) since it is easier to evaluate a model multiple times ")

print(train_dataloader, test_dataloader)

# Let's chech
print(f"DataLoader: {train_dataloader, test_dataloader} \n \n " 
      f"The length of train_dataloader {len(train_dataloader)} \n "
      f"the lenght of the test_dataloader {len(test_dataloader)}")

#Check what train_dataloader has ??
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape, train_labels_batch.shape)

# SHow a sample
torch.manual_seed(42)
DATASET_NAME = "11.png"
SAVED_DATASET = DATASET_PATH / DATASET_NAME

random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
print(len(train_features_batch), '***** ', random_idx, '888888' )
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap='gray')
plt.title(class_names[label])
plt.axis(False)
plt.savefig(SAVED_DATASET)

print(f"Image size: {img.shape} \n "
      f"Label: {label} and label size is {label.shape}")



print(f'Creating a base line model with linear layers ' * 2)

# 3. Model_0 build a base line model >>> as a starting ML model experiments ... will imporved by next models
# Start simply and add complexity if necessary.
# Creating a flatten layer >>>> are continuous range of dims into a tensor. (Sequential)
flatten_model = nn.Flatten()

# Get a single sample
x = train_features_batch[0]
print(x.shape)

# Flatten the sample
output = flatten_model(x) # perform the forward pass

print(f"Shape before flattening: {x.shape} -> [color_channels, height, width]\n "
      f"Shape after Flattening: {output.shape} [color_channels, height * width] \n "
      f"x: {x[:0]} \n "
      f"output: {output[:0].squeeze()}")


class FashinMNISTModelV0(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)


# instance of model
torch.manual_seed(42)

# Setup model with input parameters
model_0 = FashinMNISTModelV0(
    input_shape= 784, # this is 28 * 28
    hidden_units = 10, # how many nodes in the hidden layer
    output_shape = len(class_names) # one for every class
).to("cpu")

print(model_0)

# dumpy model to check the model correctly 
dummy_x = torch.rand([1, 1, 28, 28])
result = model_0(dummy_x).shape
print(result )
print(model_0.state_dict())


# 101. Creating a loss function and optimizer

# 3.1 Setup a loss function and optimizer
# Loss function : with multiclass data: nn.crossEntropyLoss()
# optimizer: torch.optim.SGD() , for stochastic gradients descent
# Evaluation metric - classificaton matric >.. accuracy

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

# Create a function to Time our experiments
# ML is very experimental >> to track >>> a model perfromance >> accuracy, loss >> and how fast it runs
# Trade off >> between fast and perfromance ML model ...

def print_train_time(start: float,
                     end:float,
                     device: torch.device = None):
    """Prints different between start and end time."""
    total_time = end - start
    print(f"Train time on {device}: took a: {total_time:.3f} seconds")
    return total_time

start_time = timer()
for i in tqdm(range(1000)):
    ...
end_time = timer() 
time_last = print_train_time(start=start_time, end=end_time, device="cpu")
print(f"The time consumed is: {time_last:.3f}")


# 3.3 Creating a training loop and training a model on bactches of data ...
# Highlight that the optimizer will update a model's parameters once per batch rather than once per epoch

print(f"Creating a training loop and training a model on batches of data:  "
      f"1. loop through epochs..."
      f"2. Loop through training batches, perfrom training steps, calculate the training loss per batch "
      f"3. Loop through the testing batches, perform testing steps, calculate the test loss per batch "
      f"4. Print out what's hapenin .. "
      f"5. Time it all for fun... ")

# tqdm for progress par ... .auto : recognize what compute environment using >> like jupyter


# Set the seed and start the timer

torch.manual_seed(42)
train_time_start_on_cpu = timer()

#Set the number of epochs ( small for faster)
epochs = 1

# Create training and test loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch} \n -------")
    ### Training
    train_loss = 0
    # Add a loop to loop through the trainig batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()
        # 1. Forward pass
        y_pred = model_0(X)

        # 2. calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # to accumalate the training loss value every batch for 823 steps

        # 3. optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        #5. optimizer steps: memory efficent , and updating our model parameters once per batch ..
        # since optimizer.step call within the batch loop, rather than the epoch loop ..
        optimizer.step()

        # Print out what's happenin
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)} / {len(train_dataloader.dataset)} samples.")

    # Divide total train loss by length of train dataloader
    train_loss /= len(train_dataloader) # because we already accumulated train loss for every batch so we need to average it accross how many batches in the dataloader

    ### Testing
    test_loss, test_acc = 0, 0
    model_0.eval()

    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            # 1. Forward Pass
            test_pred = model_0(X_test)

            # 2. calculate the test loss (accumulatively)
            test_loss += loss_fn(test_pred, y_test) 
            
            # 3. Calculate the accuracy for test 
            test_acc += accuracy_fn(y_true=y_test,
                                    y_pred=test_pred.argmax(dim=1)) # Since the raw output of the model
            # are going to be logits, but , the accuracy_fn expects that our (true labels, and predctions)
            # to be on the same format.... Since our test_pred is logits, we need using(argmax(dim=1)  to find the larger value
            # with highest index that will equavilent to the true labels. So, we can compare labels to labels.

        # Calculate the test loss average per batch
        test_loss /= len(test_dataloader) # sinc test_loss accumulated per batch, we just divide them on the length of test_dataloader

        # Calculate the test accuracy average per batch
        test_acc /= len(test_dataloader)

    # print out what's hapenin
    print(f"\n Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f} ")

# Calculate the trainig times
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(train_time_start_on_cpu,
                                            train_time_end_on_cpu,
                                            device=str(next(model_0.parameters()).device))


print(f"104 writing an evaluation function to get our model's results .. "
      f"Make predictions and get Model 0 results ... ")
device = "cuda" if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device= device
               ):
    """Returns a dictionary containing the results of model predicting on data_loader .. """
    loss, acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            X, y = X.to(device), y.to(device)
            # Make predictions
            y_pred = model(X)

            # Accumalate the loss and acc values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim=1)) # Since our y_pred (raw logits), so we need
            # to convert them to labels either using (softmax for a predction probability) OR skpping the softmax step
            # AND instead using the argmax(dim=1) to get the index where the highest value logit is. ..

        # Scale the loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(), # return it to a single value using .itme()
            "model_acc": acc }

device = "cuda" if torch.cuda.is_available() else "cpu"
print("running on: ", device)
print(str(next(model_0.parameters()).device) )
print('paramters ' * 5)

# Calculate model 0 results on test dataset
model_0_results = eval_model(model=model_0.to(device),
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn,
                             device=device)

print(model_0_results)


print(f"5. Setup device agnostic code for using GPU if there is one: cuda: {torch.cuda.is_available()} ")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("running on: ", device)

# 6. Building a better model with non-lineariy >> power of non-linearity ...
# Create a model with non-linear and linear data ...
class FashinMNISTModelV1(nn.Module):
    def __init__(self,
               input_shape: int,
               hidden_units: int,
               output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # flatten the inputs into a single vector
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape),
            nn.ReLU()
        )
    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)

print(str(next(model_0.parameters()).device), ' Model_0 is running on >>>>>>>>>>>> ')

# Create an instance of model_1
torch.manual_seed(42)
model_1 = FashinMNISTModelV1(
    input_shape = 784, # this is a output of the flatten layer after 28 * 28,
    hidden_units = 10,
    output_shape = len(class_names)).to(device)

print(str(next(model_0.parameters()).device), ' Model_1 is running on >>>>>>>>>>>> ')

print("Create a loss function and optimizer .....")
# 6.1 create a loss function, ooptimizer and evaluation metrics
from helper_functions import accuracy_fn
loss_fn = nn.CrossEntropyLoss() # measure how wrong our model is
optimizer = torch.optim.SGD(params= model_1.parameters(), # update our model parameters to reduce the loss
                            lr=0.1)


print("Functionizing training and testing/evaluation loops ..... "*2)
print(f"Let's create a function for: "
      f"training loop - training_step()"
      f"testing loop - test_step()")

def train_step(model: torch.nn.Module,
               data_loader: DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: optimizer,
               accuracy_fn,
               device: torch.device=device):
    """Performs training steps with model trying to learn on data_loader"""
    train_loss, train_acc = 0, 0

    # Add a loop to loop through the trainig batches
    #Put model on the target device
    model.train()


    for batch, (X, y) in enumerate(data_loader):

        #Put data in the target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass (outputs the raw logits from the model)
        y_pred = model(X)

        # 2. calculate loss and accuracy (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss  # to accumalate the training loss value every batch for 823 steps
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # go from logits to prediction labels
        # 3. optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. optimizer steps: memory efficent , and updating our model parameters once per batch ..
        # since optimizer.step call within the batch loop, rather than the epoch loop ..
        optimizer.step()

    # Divide total train loss and acc by length of train dataloader
    train_loss /= len(data_loader)  # because we already accumulated train loss for every batch so we need to average it accross how many batches in the dataloader
    train_acc /= len(data_loader)
    # print out what's happenin
    print(f"Train loss: {train_loss:.4f}, train_acc: {train_acc:.2f}% \n")


def test_step(model: torch.nn.Module,
              data_loader:DataLoader,
              loss_fn:torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):

    """returns a testing loop steps on a model going over data_loader .... """
    ### Put model eval
    model_0.eval()
    test_loss, test_acc = 0, 0

    # Turn on inference_mode context manager / predictions/ testing
    with torch.inference_mode():
        for X, y in data_loader:
            # send data to the target device
            X, y = X.to(device), y.to(device)

            # 1. Forward Pass (outputs raw logits)
            y_pred = model(X)

            # 2. calculate the test loss and test accuracy (accumulatively)
            test_loss += loss_fn(y_pred, y)
            # 3. Calculate the accuracy for test
            test_acc += accuracy_fn(y_true=y, # go from logits to pred probs
                                    y_pred=y_pred.argmax(dim=1))  # Since the raw output of the model
            # are going to be logits, but , the accuracy_fn expects that our (true labels, and predctions)
            # to be on the same format.... Since our test_pred is logits, we need using(argmax(dim=1)  to find the larger value
            # with highest index that will equavilent to the true labels. So, we can compare labels to labels.

        # Calculate the test loss and test acc average per batch
        test_loss /= len(data_loader)  # sinc test_loss accumulated per batch, we just divide them on the length of test_dataloader
        test_acc /= len(data_loader)
        # print out what's hapenin
        print(f"\n Test loss: {test_loss:.4f} | test accuracy: {test_acc:.2f}% \n")


print(f" Model 1: Training and testing with our training and testing functions ... \n")

torch.manual_seed(42)

# Measure time
from timeit import default_timer as timer
train_time_start_on_gpu = timer()

# Set epcohs
epochs = 1

# Create a optimizations and evaluation loop using train_step() and test_step()

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch} \n ----------------- ")
    train_step(model=model_1,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)

    test_step(model=model_1,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)

train_time_end_on_gpu = timer()

total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=device)


print("# Train time on CPU , maybe faster than GPU depending on data/hardware becuase the overhead/ copying "
      "data/model to and from GPU outweighs the compute benefits of offred by GPU "
      "2. The hardware CPU outperform GPU ... ")
print(total_train_time_model_0)
# print(total_train_time_model_1)
#
# # Get model_1 result dictionary
#
# model_1_results = eval_model(model=model_1,
#                              data_loader=test_dataloader,
#                              loss_fn=loss_fn,
#                              accuracy_fn=accuracy_fn,
#                              device=device)
# print(model_0_results)
# print(model_1_results)