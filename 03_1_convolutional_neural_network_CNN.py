print(f"# Model 2: Building a convolutional neural network."
      f"CNN's are also known as ConvNets."
      f"CNN's are known for capability to find patterns of visual data. "
      f"CNN hyperparameters: "
      f"1. Input layer "
      f"Conv layer "
      f"3. Hidden activation "
      f"4. Pooling layer "
      f"5. output layer ?? Linear layer ")


# Typical CNN architecture.
print(f"Input image >> preprocessing: RGB image , Conv layer > ReLU layer, Pooling layer "
      f"Researchers coming our almost everyday ::: how to best construct these layers ... as they can be combined differently"
      f"How to get input >>> output ... Linear output layer .. classes "
      f"Deeper CNN >>> more layers >> the more layers to add to NNN the better patterns it captures ... "
      f"each subsequent layer receives the input from previous layer ... except the resdual connection ... ")

print(f" 113: Model 2: Coding CNN " * 3)
from computer_vision_workthrough import *
import torch
import random
from torch import nn
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor # Convert a PIL image or numpy.ndarray to tensor
from pathlib import Path
import pandas as pd
from helper_functions import *
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm.auto import tqdm
import torchmetrics, mlxtend
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

from helper_functions import accuracy_fn


#Check versions
print(torch.__version__)
print(torchvision.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32



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


#Turn dataset into iterabel
train_dataloader = DataLoader(dataset = train_data,
                              batch_size= BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(dataset = test_data,
                             batch_size= BATCH_SIZE,
                             shuffle=False)


class_names = train_data.classes
print(class_names)

class_to_idx = train_data.class_to_idx
print(class_to_idx)
targetss = train_data.targets
print(targetss)

# see the first training examples of ToTensor >>>> convert a PIL in numpy array to Tensor >> of shape (C H W) in range [0.0, 1.0] 
image, label = train_data[1]
#print(image, '\n', label)

# recall train/test steps functions 
def train_step(model: torch.nn.Module,
               data_loader: DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn: accuracy_fn,
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
    model.eval()
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

def print_train_time(start: float,
                     end:float,
                     device: torch.device = None):
    """Prints different between start and end time."""
    total_time = end - start
    print(f"Train time on {device}: took a: {total_time:.3f} seconds")
    return total_time


def eval_model(model: torch.nn.Module,
               data_loader: DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device= device
               ):
    """Returns a dictionary containing the results of model predicting on data_loader .. """
    loss, acc = 0, 0
    model.eval()
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


# Create a convolutional neural network
class FashinMNISTModelV2(nn.Module):
    """Model architecture that replicate a tinyVGG model from CNN explainer website"""
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=(3,3),
                      stride=(1),
                      padding=1), # Values we can set ourself in NN are called hyperparameters
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=(1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7*7, # Since the role of matrix multiplication is the inner dimensions have to match
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        #print(f"Output shape of conv_block_1: {x.shape}")
        x = self.conv_block_2(x)
        #print(f"Output shape of conv_block_2: {x.shape}")
        x = self.classifier(x)
        #print(f"Output shape of Classifier: {x.shape}")
        return x



#INstantiate a model
torch.manual_seed(42)
model_2 = FashinMNISTModelV2(input_shape=1,
                             hidden_units=10,
                             output_shape=len(class_names)).to(device)




print(f"Stepping through nn.Conv2d  "
      f"out = biase * i + sum (weight , c,k * X)"
      f"Input: (N, Cin, Hin, Win) or (Cin, Hin, Win) "
      f"Output: (N,Cout, Hout, Wout) or (Cout, Hout, Wout) where:"
      f"Hout = =Hin + 2 * padding[0] - dilation[0] * (kernel_size[0] -1) -1 / stride[0] +1 "
      f"Wout = Win + 2 * padding[1] - dilation[1] * (kernel_size[1] -1 ) -1 / stride[1] +1)")

torch.manual_seed(42)

# Create a batch of images
images = torch.randn(size=(32, 3, 64, 64))
test_image = images[0]#.unsqueeze(0)
print(f"image batch shape: {images.shape}")
print(f"single image  shape: {test_image.shape}")

#print(model_2.state_dict())

# Create a single conv2d layer
torch.manual_seed(42)
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=3,
                       stride=1,
                       padding=1)

print(conv_layer)
# Pass the data throught the convolutional layer 
conv_output = conv_layer(test_image.unsqueeze(0)) # test_image.unsqueeze(0)
print("output conv layer " * 3)
#print(conv_output)
print(conv_output.shape)


print(f"Stepping throught MaxPool2D: output value of the layer input  (NCHW, output: NCHW, K-size=(2,2)")
#print out hte original image shape with out unsqueezed dimenitn 
print(f" Test image original shape: {test_image.shape}")
print(f" Test image with unqueezed dimension: {test_image.unsqueeze(0).shape}")

# Create a sample nn.MaxPool2d layer
max_pool_layer = nn.MaxPool2d(kernel_size=2)

#Pas data throught just conv_layer
test_image_through_conv = conv_layer(test_image.unsqueeze(dim=0))
print(f" Shape after going throught conv_layer(): {test_image_through_conv.shape}")

#Pass data throught the max pool layer 
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
print(f" Shape after going through conv_Layer and max_pool layer(): {test_image_through_conv_and_max_pool.shape}")

print(test_image.shape)

# create a randome tensosr with a similar number of dimensions to our images 
torch.manual_seed(42)
random_tensor = torch.randn(size=(1, 1, 2, 2,))
print(f"\nRandom tensor: \n {random_tensor} ")
print(f"\nRandom tensor shape: \n {random_tensor.shape} ")

# create a max pool layer
max_pool_layer = nn.MaxPool2d(kernel_size=2)

# passs the ranodm tensor throught the max pool layer
max_pool_tensor = max_pool_layer(random_tensor)

#print(f"n Max pool tensor:\n {max_pool_tensor}")
print(f"Max pool tensor shape: {max_pool_tensor.shape}")

print(f"in_channel: grayscale/RGB, out_channel: defined, kernel_size: determined, stride: d, padding: same,valid).")


DATASET_PATH = Path("visualization")
DATASET_PATH.mkdir(parents=True, exist_ok=True)
IMAGE_NAME = "image.png"
SAVED_IMAGE = DATASET_PATH / IMAGE_NAME
plt.imshow(image.squeeze(), cmap='gray')
plt.savefig(SAVED_IMAGE)

# create a random tensor ... 
random_tensor = torch.randn(size=(1,28,28))
print(random_tensor.shape)

# Pass image to a model 
model_2(random_tensor.to(device).unsqueeze(dim=1))


        

print(f"Setting up the loss function and the optimizer" * 3)
### 7.3 : loss fun/eval metrics/opotimizer

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.1)
print(optimizer, "\noptimoptimoptimoptimoptimoptimoptimoptimoptim"*3)

# 4.7 Training and testing functions using our traiing and testing funcitons
torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs= 3
# Measure time 
from timeit import default_timer as timer
train_time_start_model_2 = timer()
#Train and test model

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch} \n ------")
    train_step(model=model_2,
               data_loader=train_dataloader,
               loss_fn = loss_fn,
               optimizer = optimizer,
               accuracy_fn = accuracy_fn,
               device = device)

    test_step(model=model_2,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)

train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(start=train_time_start_model_2,
                                            end=train_time_end_model_2,
                                            device=device)


# Get results dictionary
model_2_results = eval_model(
 model= model_2,
 data_loader = test_dataloader,
 loss_fn=loss_fn,
 accuracy_fn=accuracy_fn,
 device=device

)

print(model_2_results)

## 8. Compare model results and training time ...

compare_results = pd.DataFrame([model_0_results,
                                model_1_results,
                                model_2_results])

print(compare_results)


# Add training time to results comparing ...
compare_results["training_time"] = [total_train_time_model_0,
                                    total_train_time_model_1,
                                    total_train_time_model_2]


DATASET_PATH = Path("visualization")
DATASET_PATH.mkdir(parents=True, exist_ok=True)
IMAGE_NAME = "compare_results.csv"
SAVED_IMAGE = DATASET_PATH / IMAGE_NAME

print(compare_results)
compare_results.to_csv(SAVED_IMAGE)


# Visualize our model results ................
compare_results.set_index("model_name")["model_acc"].plot(kind='bar')
plt.xlabel("accuracy (%)")
plt.ylabel("model")
plt.savefig('/home/mhamdan/DeepLearning_PyTorch_2022/visualization/compare_results_bar.png')


## Make and evaluate random predictions with best model ...


def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # add a batch dimension and send it to device
            sample = torch.unsqueeze(sample, dim=0).to(device)

            # Forward pass (model outputs raw logits)
            pred_logit = model(sample)

            # Get the prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            # Get pred prob off the GPU for further calculations
            pred_probs.append(pred_prob.cpu())

    # Stack the pred_prob to turn a list into a tensor
    return torch.stack(pred_probs)

random.seed(42)
test_samples = []
test_labels = []

for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

    # View the first sample shape
print(test_samples[0].shape)


IMAGE_NAME = "ranomdSample.png"
SAVED_IMAGE = DATASET_PATH / IMAGE_NAME
plt.imshow(test_samples[0].squeeze(), cmap='gray')
plt.title(class_names[test_labels[0]])
plt.savefig(SAVED_IMAGE)



#### Make predictions ..

pred_probs = make_predictions(model=model_2,
                              data=test_samples)


# View first two predictions probabilities
print(pred_probs[:2])

# Convert prediction probabilities to labels
pred_classes = pred_probs.argmax(dim=1)
print(pred_classes)
print('\n VS original labels \n')
print(test_labels)


print(f" Plotting our best model predictions on the test set and evaluating them \n"
      )

# plot prediction
plt.figure(figsize=(9,9))
nrows = 3
ncols = 3

for i, sample in enumerate(test_samples):
    # create subplot
    plt.subplot(nrows, ncols, i +1)
    # Plot the target images
    plt.imshow(sample.squeeze(), cmap='gray')

    # Find teh prediction (in text form, eg. 'Sandal'
    pred_label = class_names[pred_classes[i]]

    # Get the truth label in text form
    truth_label = class_names[test_labels[i]]

    # Create a title for the plot
    title_text = f"Pred: {pred_label} | Truth: {truth_label}"

    # Check for a quality between pred and truth and change color of the title text
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, color='green') # green text if predictions same as truth
    else:
        plt.title(title_text, color='red')

    plt.axis(False)
    plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/visualization/predictionsVStruth.png")



print(f"Confusion matrex... "* 3)
# Making a confusion matrix for further prediction evaluation ... classification model visually

print(f" Make predictions with our trained model on the test dataset "
      f"2. Make a confusion matrix 'torchmetrics.ConfusionMatrix' "
      f"3. Plot the confusion matrix using 'mlextend.plotting.plot_confusionmatrix)")

# Import tqdm.auto import tqdm

# Make prediction with trained model
y_preds = []
model_2.eval()

with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Makeing predictions ........"):
        # Send data and target to device
        X, y = X.to(device), y.to(device)
        # Forward pass (raw logits
        y_logit = model_2(X)

        # Turn predictions from logits -> prediction probabilities -> prediction label
        y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)

        # Put predictions on cpu for evaluation
        y_preds.append(y_pred.cpu())

# Concatenate list of predictions into a tensor
print("list \n", y_preds[:10], len(y_preds) )#, 'its shape \n',y_preds.shape)
y_pred_tensor = torch.cat(y_preds)
print("tensor \n", y_pred_tensor[:10])
print("length tensor \n", len(y_pred_tensor), 'its shape \n',y_pred_tensor.shape)

print(mlxtend.__version__, ' __version__ of mlxtend ....... evaluation with confusion matrix ')


# Set up Confusion Matrix instance and compare predictions to target
confmat = ConfusionMatrix(num_classes=len(class_names))

confmat_tensor = confmat(preds=y_pred_tensor,
                         target= test_data.targets)


#3. Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat = confmat_tensor.numpy(), # matplotlib. works wiht numpy
    class_names = class_names,
    figsize=(10, 7)
)
fig.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/visualization/ConfusionMatrix.png")
print(f" Notes on ConfusionMatrix: "
      f"1. The ideal confusion matrix will have all the diagnol rows darken, "
      f"2. All the values are representing diagnolly ... and no values on the upper and bottom triangle."
      f"3. That means the predicted labels (x-axis) are lined with the true labels (y-axix)     "
      f"4. some mispredicted cases, for example when the true label is T-shired/top the model predicts as shirt "
      f"As seen the model confused to predict Pullover when the actual labels is Coat and so on ")


print("Save model and loaded " * 4)
# Save and load the best performing model

# create model directory path ..
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)
# Create a model save name
MODEL_NAME = "03_1.pytorch_computer_vision_model_2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state_dict
print(f" Saving model to: {MODEL_SAVE_PATH} ")
torch.save(obj=model_2.state_dict(),
           f=MODEL_SAVE_PATH)


# Create a new instance of our model 2 .>> to load the state_dict() with same parameters as our original model
torch.manual_seed(42)
loaded_model_2 = FashinMNISTModelV2(input_shape=1,
                                    hidden_units=10,
                                    output_shape=len(class_names))

# Loading teh saved state_dict()
loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Send teh model to the target device
print(loaded_model_2.to(device))


# Evaluate teh model is almost as training 
print(model_2_results)

torch.manual_seed(42)
loaded_model_2_results = eval_model(
    model=loaded_model_2,
    data_loader = test_dataloader,
    loss_fn = loss_fn,
    accuracy_fn = accuracy_fn

)
print("loaded model resutls "* 8)
print(loaded_model_2_results)


# check if model resutls are close to each other
are_they_close = torch.isclose(torch.tensor(model_2_results["model_loss"]),
              torch.tensor(loaded_model_2_results["model_loss"]),
              atol = 1e-02) # adjust the tolerance level with 2 dicimal points

print(are_they_close)