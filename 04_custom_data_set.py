print(f"What is a custom dataset >>> "
      f"build your own datasets ... to make them compatible wiht PyTorch ... "
      f"classify >> image classifcation problem"
      f"Text ... spam not spam "
      f"Any type of vision data .... "
      f"any text data .. torch text"
      f"torchAudio for audio provm "
      f"torchRec  fro recommended systems "
      f"Data loading ... "
      f""
      f""
      f"TorchVison torchvision.datasets "
      f"torchData ................ will updated over time "
      f"Torchvison "
      f"TorchText for"
      f"torchAudio"
      f"vision text audion recommendsystem bonus .. TorchData "
      f"Load data ... Build image classifiction model... classify image ")

print("cover broadly ... "* 3)
print(f"1. Getting custom dataset with PyTorch "
      f"2. Becoming one with the data (preparing and visualizaion)"
      f"3. Transforming data for use with a model "
      f"4. Loading custom data with pre-built functions and custom functions  "
      f"5. Building FoodVision Mini to classify food iamges"
      f"6. Comparing models with and without data augmentation "
      f"7. Making prediction on custom data")

print('custom dataset '* 5)

print(f"Domain libraries: "
      f"1. Vision "
      f"2. text"
      f"3. audio"
      f"recommendation system "
      f"5. and so on")

import torch
import torchvision
from torchinfo import summary
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import random
import pandas as pd
from PIL import Image
from timeit import default_timer as timer
from typing import Tuple, Dict, List # put typing when creating


random.seed(42)

# Note: PyTorch 1.10.0 + is required for this course
print(torch.__version__)

# Set up device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Get data >> a subset of Food101.. has 100 classes of
# 750 training and 250 testing . starts with 3 classes and only 10% of the images
# Start ML projects, its important to try things on asmall scale and then increase the scale
# and increase the scale, to speed up how fast you can experiment.... increase the rate of experiment ..

#

print("Becoming one with data ... \n "
      "If I had 8 hours to build a machine learning model, I'd spend the first 6 hours preparing my dataset ")


import os
def walk_through_dir(dir_path):
      """Walks thruogh dir_path return its contenst """
      for dirpath, dirnames, filenames in os.walk(dir_path):
            print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}' .")
walk_through_dir("/home/mhamdan/DeepLearning_PyTorch_2022/data")



DATASET_PATH = Path("train_test")
DATASET_PATH.mkdir(parents=True,
                  exist_ok=True)

# # setup train and testing paths
# train_dir = DATASET_PATH / "train"
# test_dir = DATASET_PATH / "test"
# print(train_dir, '\n', test_dir)

# Visualizing a random data
print(f"1. Get all of the image pahts "
      f"2. Pick a radom image path using Pythons's random.choice() "
      f"Get the image class name using 'pathlib.Path.parent.stem' "
      f"Since we are working with images, open using PIL "
      f"5. Show image and print metadata")


DATA_PATH = Path("data/")
IMAGE_PATH = DATA_PATH / "pizza_steak_sushi_20_percent"
# get all image paths
image_path_list = list(IMAGE_PATH.glob("*/*/*.jpg"))
print('\n \n ')
print(len(image_path_list))
# setup train and testing paths
train_dir = IMAGE_PATH / "train"
test_dir = IMAGE_PATH / "test"
print(train_dir, '\n', test_dir)
# pick a random image path
random_image_path = random.choice(image_path_list)
print(random_image_path)

# 3/ Get the image class from the path name >> is the name of the directory where the image is stored
image_class = random_image_path.parent.stem
print(image_class)

#4. Open image
img = Image.open(random_image_path)

# 5. print metadata
print(f"Random image path: {random_image_path} \n "
      f"Image class: {image_class} \n "
      f"Image height: {img.height} \n"
      f"Image width: {img.width}")
img.save("/home/mhamdan/DeepLearning_PyTorch_2022/visualization/imgPIL.png")


# Visualizing an image with matplotlib
# Turn image to array
img_as_array = np.asarray(img)

#plot image with matplotlib
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | image shape: {img_as_array.shape} -> hight, width , color channel")
plt.axis(False)
plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/visualization/matplotlibLIP.png")
print(img_as_array)

print(f" Transforming data >>> \n "
      f"1. Before we can use our image data with PyTorch : "
      f"\n1. Turn the data into tensors (a numberical representation of our images \n "
      f"2. Turn it into a 'torch.utils.data.Dataset' and \n "
      f"subsequently >into > 'torch.utils.data.DataLoader' , we call these 'Dataset' and 'DataLoader'."
      f" \n\n  3.1 Transforming data with torchvision.transform")

# write a transofrm for image
data_transform = transforms.Compose([
      # resize our images to 64 * 64
      transforms.Resize(size=(64, 64)),
      # Flip images randomly in the horizontal
      transforms.RandomHorizontalFlip(p=0.5), # probability > 50 % if of the time if the image goes to filenames
      #Turn image into a torch.tensor
      transforms.ToTensor()
      ])

print(data_transform(img), '\n',
      data_transform(img).shape,
      data_transform(img).dtype,
      )

# Visualizing our tansformed image : tranforms prepare images to tensors... and do data augmentation.

def plot_transformed_images(image_paths, transform, n=3, seed=None):
      """Selects random images from a path of images,
      loads/transforms them then plots original imgs vs transforms version
      """
      if seed:
            random.seed(seed)
      random_image_paths = random.sample(image_paths, k=n)
      for image_path in random_image_paths:
            with Image.open(image_path) as f:
                  fig, ax = plt.subplots(nrows=1, ncols=2)
                  ax[0].imshow(f)
                  ax[0].set_title(f"Original\nSize: {f.size}")
                  ax[0].axis(False)

                  # Transform and plot target image
                  transformed_image = transform(f).permute(1,2,0) #Change order for matplotlib from(C,H,W) -> (H,W,C)
                  ax[1].imshow(transformed_image)
                  ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}")
                  ax[1].axis(False)

                  fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
                  #fig.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/visualization/testPLOTPLIB.png")
            fig.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/visualization/plot_transformed_images.png")
                  
                  
plot_transformed_images(image_paths=image_path_list,
                        transform=data_transform,
                        n=3,
                        seed=42
                        )
print("*"*100)
print("Loading image data using ImageFolder"* 4)
print(f" Load image classification data using torchvision.dataset.ImageFolder")
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,# a transform for teh data
                                  target_transform=None) # a transforms for the target/label

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

print(train_data, test_data)

# Get classes
class_names = train_data.classes
print(class_names)

# Get class names as a dictionary
class_dict = train_data.class_to_idx
print(class_dict)

# Check the lengths of dataset
print(len(train_data), len(test_data))
print(train_data.samples[0])


# INdex on the training data dataset to get an image and the label ...
img, label = train_data[0][0] , train_data[0][1]
print(img.shape, label, class_names[label])

print(f"IMage tensor :\n {img[:5]},"
      f"\n Image Shape \n{img.shape}"
      f"\n Image Data type: \n {img.dtype} "
      f"\n Image label : \n {label}"
      f"\n Label Datatype: \n {type(label)}")

# Rearrange the order of dimension
img_permute = img.permute(1, 2, 0)

#print out differn shapes
print(f"Original shape: {img.shape} -> [color_channels, height, width]")
print(f"Image permute shape: {img_permute.shape} -> [ height,width, color_channels]")

# PLot the image
plt.figure(figsize=(10, 7))
plt.imshow(img_permute)
plt.axis('off')
plt.title(class_names[label], fontsize=16)
plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/visualization/img_permute.png")


### Turn datset to DataLoader
print("Dataset to data_loader"* 3)
print("A dataloader turn our dataset to iteratable and customize the batch_size so our model can see batch_size images at each time")
# Turn train and test dataset to data loader
batch_size = 32
train_data_loader = DataLoader(dataset=train_data,
                               batch_size=batch_size,
                               num_workers=os.cpu_count(),
                               shuffle=True)
test_data_loader = DataLoader(dataset=test_data,
                              batch_size=batch_size,
                              num_workers=os.cpu_count(),
                              shuffle=False)

print(train_data_loader, '\n', test_data_loader)
print(len(train_data_loader), '\n', len(test_data_loader))

img, label = next(iter(train_data_loader))

# Batch size will now ber 32, change img_permutep
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape} -> [add one dimension]")

print(f"5.2 Loading image data with a custom 'Dataset' " * 4)
print(f"1. Load images from file\n"
      f"2. get class names from data sets \n"
      f"3. get class as dictionary from data set.. ")

print(f"Pros: "
      f"1. create a 'Dataset' out fo almost anything \n "
      f""
      f"2. limite to PyTorch pre-built 'dataset' functions \n "
      f""
      f"cons: "
      f"might not work or results to error ...")



print(train_data.classes, train_data.class_to_idx)

# 5.1 creating a hellper function to get class names
print(f"1. get the class names using 'os.scandir()' to traverse a target directory ("
      f"ideally the directory is in standard imge classification format "
      f"\n2. Raise an error if the class names aren't found (if this happens , there might be something "
      f"wrong with directory structure"
      f"\n3. Turn the class name into a dict and a list and return them  ")

# setup the path for the target directory
target_directory = train_dir

print(f"target dir: {target_directory}")

# Get the class names from the target directory

class_names_founds = sorted(entry.name for entry in list(os.scandir(target_directory)))
print(class_names_founds)
print(list(os.scandir(target_directory)))

def find_classes(directory:str) -> Tuple[List[str], Dict[str, int]]:
      """Finds the class folder names in a target directory .  """
      #1. Get the class names by scanning the target directory
      classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

      #2. Raise an error if class names cannot be found
      if not classes:
            raise FileNotFoundError (f"Couldn't find any classes in {directory} ... please check the file structure.")

      #3. Create a dictionary of index label (computers perefer numbers rather than strings as labels)

      class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

      return classes, class_to_idx

print(find_classes(target_directory))


print(f" \n Create a custom dataset to replicate 'ImageFolder' "
      f"\n all datasets map from keys (targets) to data samples should subclass it. "
      f"\n overwirite  __getitem__() to get a sample ... for a given key like 100  "
      f"__len__()  to get the size dataset  ... ")


print(f"TO create our own custom dataset: "
      f"\n1. Subclass 'torch.utils.data.Dataset'"
      f"\n2. Init our subclass with a target direcotry ( the directory we'd like to get data from)"
      f"as well as a transform if we'd lik to transfor our data."
      f"\n3 Create several attributes: "
      f"\na Paths - paths images "
      f"\nb transform - if needed "
      f"\nc classes - a list of the target classes "
      f"\nd class_to_idx - a dict of the target classes mapped to integer labels "
      f"\n \n 4. Create a function to load_images(), this function will open an images"
      f"\n 5. Overwrite the __len()__ method to return the length of our dataset "
      f"\n6 overwrite the __getitem()__ method to return a given sample when passed an index ")

# write a custom dataset class
#1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
      #2. Initialize our custom dataset 
      def __init__(self,
                   target_dir: str,
                   transform=None) ->None:

            #3. Create class attributes / get all the image paths
            self.paths = list(Path(target_dir).glob("*/*.jpg"))
            #setup transforms
            self.transform = transform
            #Create class/ class_to_idx
            self.classes, self.class_to_idx = find_classes(target_dir)



      # 4. Load image
      def load_image(self, index:int) -> Image.Image:
            "Opens an image via a path and returns it."
            image_path = self.paths[index]
            return Image.open(image_path)

      #5. Overwrite the __len__()
      def __len__(self) -> int:
            "return the total number of samples "
            return len(self.paths)

      #6. Overwrit __getitem__() to return a particular sample
      def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
            "Returns one sample of data, data and label (X, y) ... "
            img = self.load_image(index)
            class_name = self.paths[index].parent.name # expect paht in format: data_folder/class_name/image.jpg
            class_idx = self.class_to_idx[class_name]
            if self.transform:
                  return self.transform(img), class_idx # return data, label (X,y)
            else:
                  return img, class_idx # return untransformed image and label



print(f"Comparing our custom dataset to Pytorch built in .."
      )
# Create a transform
SIZE = (64, 64)
probability = 0.5
train_transforms = transforms.Compose([
      transforms.Resize(size=SIZE),
      transforms.RandomHorizontalFlip(p=probability),
      transforms.ToTensor()
])

test_transforms = transforms.Compose([
      transforms.Resize(size=SIZE),
      transforms.ToTensor()
])
print(train_dir)
# Test out ImageFolderCustom
train_data_custom = ImageFolderCustom(target_dir=train_dir,
                                      transform=train_transforms
                                      )
test_data_custom = ImageFolderCustom(target_dir=test_dir,
                                     transform=test_transforms)

print(train_data_custom, test_data_custom)
print(len(train_data_custom), len(test_data_custom))

print(train_data_custom.class_to_idx)

# Check for equality between original ImageFolder dataset and our customization
print(train_data_custom.classes == train_data.classes)
print(test_data_custom.classes == test_data.classes)

print("Create a function to display raondom images ..... ")
print(f"1. Take in 'Dataset' and a number of other parameters such as class names and how mnay images to Visualize \n "
      f"2. To prevent the display getting out of hand, let's cap the number of images to see at 10 \n "
      f"3. Set the random seed for reproducibility \n "
      f"4. Get a list of random sample indexes from the target dataset \n "
      f"5. Setup a matplotlib plot \n "
      f"6. Loop throght the random sample images and plot them with matplotlig "
      f"\n7. Make sure teh dimensions of our images line up  with matplotlib HWC \n" )


#1. Create a function to take in a Dataset
def display_random_images(dataset: Dataset,
                          classes: List[str]= None,
                          n: int=10,
                          display_shape:bool=True,
                          seed: int=None):
      #2. Adjust display if n is too high
      if n > 10:
            n = 10
            display_shape = False
            print(f"For display, purposes, n shouldn't be larger than 10, and removing shape display. ")


      # 3. set the random seed for reproducibility
      if seed:
            random.seed(seed)

      #4. get some random sample indexes
      random_samples_idx = random.sample(range(len(dataset)), k=n)

      #5. Setup plot
      plt.figure(figsize=(16, 8))


      # 6. loop through random indexes and plot them using matplotlib
      for i, targ_sample in enumerate(random_samples_idx):
            targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

            # 7. Adjust tensor dimensions for plotting/home/mhamdan/DeepLearning_PyTorch_2022/visualization
            targ_image_adjust = targ_image.permute(1, 2, 0) # [color_channels, height, width] -> [height, width, color_channels]""

            # Plot adjusted samples
            plt.subplot(1, n, i+1)
            plt.imshow(targ_image_adjust)
            plt.axis('off')
            if classes:
                  title = f"Class: {classes[targ_label]}"
                  if display_shape:
                        title = title + f"\nshape: {targ_image_adjust.shape}"
                  plt.title(title)
            plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/visualization/CustomDatasetVis_at_end.png")


# display random images from ImageFolder creared datast
display_random_images(train_data,
                      n=5,
                      classes=class_names,
                      seed=None)

# display random images from custom ImageFolder creared datast
display_random_images(train_data_custom,
                      n=20,
                      classes=class_names,
                      seed=42)



print(f"5.4 Turn custom loaded images into 'DataLoader'")

train_dataloader_custom = DataLoader(dataset=train_data_custom,
                                     batch_size=batch_size,
                                     num_workers=os.cpu_count(),
                                     shuffle=True)

test_dataloader_custom = DataLoader(dataset=test_data_custom,
                                    batch_size=batch_size,
                                    num_workers=os.cpu_count(),
                                    shuffle=False)
print(f"train_dataloader_custom * test_dataloader_custom is \n {train_dataloader_custom, test_dataloader_custom}")

# Get image and label for custom dataloader
img_custom, label_custom = next(iter(train_dataloader_custom))
print(img_custom.shape, label_custom)

SEIZE = (224, 224) 
print(f" dataaugmentation via torchvision.transform .... "* 3)
print(f" Other fold of transform ... data augmentation ... \n Data augmentation is the process of artificially adding diversity to the training data. \n In case of image data: image transformations into the training data.. rotate. shift, zoom, croping, rotatin, shearing \n Let's take a look and one particular type of data augmentation used to dtrain PyTorch visiion models to state of the art levels.... \n to increase the diversity of the trainig data ... so that the images become harder to learn and therefore, any real world images can be reconize by the model \n torchvision permitives ... \n ResNet50 ... accurcy \n \n Lets look at trivialAugmentWide  ")
train_transform = transforms.Compose([
                    transforms.Resize(SIZE),
                    transforms.TrivialAugmentWide(num_magnitude_bins=31),
                    transforms.ToTensor()
                                      

])

test_transform = transforms.Compose([
                  transforms.Resize(SIZE),
                  transforms.ToTensor()
])

# Get all teh image paths
image_path_list1 = list(IMAGE_PATH.glob("*/*/*.jpg"))
print(image_path_list1[:3])

# PLot random transformed iamges
plot_transformed_images(
                        image_paths = image_path_list1,
                        transform=train_transform,
                        n = 3,
                        seed = None
              )




print(f"Model0: Tiny VGG model withouth data augmentation ... \n 1. replicate the TinyVGG architecture from CNN Explainer \n 1. Creating Transforms and loading data for Model 0")

#Create simple transform 
simple_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# 1. Load and transform data
train_data_simple = datasets.ImageFolder(root=train_dir,
                                         transform=simple_transform)
test_data_simple = datasets.ImageFolder(root=test_dir,
                                        transform = simple_transform)

#2. Turn the data set into DataLoader
BATCH_SIZE = 8
NUM_WORKERS = os.cpu_count()

# Create DataLoaders
train_dataloader_simple = DataLoader(dataset=train_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=NUM_WORKERS)
test_dataloader_simple = DataLoader(dataset=test_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=NUM_WORKERS)

print(train_dataloader_simple, test_dataloader_simple) 
test_sample, test_target = next(iter(test_dataloader_simple))
print(test_sample.shape, test_target.shape)


# Tiny VGG architecture .... 
print(f" TinyVGG architecture ... 7.2 ... " *2)


class TinyVGG(nn.Module):
    """ Model architecture copying from CNN explainer """
    def __init__(self, input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__() 
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # defualt stride value is the same as kernel size
            
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # defualt stride value is the same as kernel size
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features= hidden_units * 13 * 13,
                      out_features = output_shape)
        )
        
     
            
    def forward(self, x):
        x = self.conv_block_1(x)
        #print(x.shape)
        x = self.conv_block_2(x)
        #print(x.shape)
        x = self.classifier(x)
        #print(x.shape)
        return x
        #return self.classifier(self.conv_block_2(self.conv_block1(x))) # Operator fusion ... making deep learnig Brr 
        
        
# Create an instance model

torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(class_names)).to(device)

print(model_0)


print(f"7.3 Try a forward pass in a single image ")
# get a singel image
image_batch, label_batch = next(iter(train_dataloader_simple))
print(image_batch.shape, label_batch.shape)

# Try a forward pass
model_0(image_batch.to(device))
print(model_0.state_dict())

print(f"torchinfo import summary -> used to display model information ... ")
print(summary(model_0, input_size=[1, 3, 64, 64]))


print(f" training and test steps " * 3 )
# 7.5 create train and test loops functions 

print(f"'train_step()' takes in a model and dataloader and trains teh model the dataloader \n 'test_step()' takes in a model and dataloader and evalualtes the mdoel on the dataloader ... ")

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer:torch.optim.Optimizer,
              device=device):
    # put the model in the train mode
    model.train()
    
    # Setup train loss and train accuracy 
    train_loss, train_acc = 0, 0
    
    #Loop through data laoder data batches.
    
    for batch, (X, y) in enumerate(dataloader):
        #Send data to the target device
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        y_pred = model(X)
        
        # 2. Calculate the loss and accumulate accross all batches
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        
        #Optimize zero grad
        optimizer.zero_grad()
        
        #4. Loss backward
        loss.backward()
        
        #5. Optimizer step
        optimizer.step()
        
        # Cacluate and accumulate the accuracy metric accross all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc +=(y_pred_class==y).sum().item()/len(y_pred)
        
    # Adjust matrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


# test step create .... 
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device=device):
    # Put model in eval mode
    model.eval()
    
    # Setup test loss and test accuracy values 
    test_loss, test_acc = 0, 0
    
    # Turn on inference mode 
    with torch.inference_mode():
        # Looop through DataLoader batches ... 
        for batch, (X,y) in enumerate(dataloader):
            # Send data to the target device 
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred_logits = model(X)
            
            # 2. Calculate  and accumulate the loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate the accuracy 
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc 
        
        

print("Create a function Train ................. "*3)

# Create a train function combine a train_step() and test_step()

#1. create train function that takes in various model parameters + optimizer + dataloader  + optimizer + loss function 

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int =5,
          device = device):
    
    #2. Create empty results dectionary of [str: List[float]]
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
              }
    
    #3. Loop through training and testing steps for a number of epochs 
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model,
                                           dataloader=train_dataloader,
                                           loss_fn = loss_fn,
                                           optimizer = optimizer,
                                           device =device)
        test_loss, test_acc = test_step(model,
                                        dataloader=test_dataloader,
                                        loss_fn = loss_fn,
                                        device=device)
        
        
        
        
        # Print out what's hapenin
        print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test acc: {test_acc:.4f} ")
        
        
        # Update results dictionary 
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        
    #6. Return teh filled results at eh end of the epoch 

    return results  

        

# Train model .... 
print("Train a model " * 5 )

# Set the random seeds 
torch.manual_seed(42)
torch.cuda.manual_seed(42) 

# Set number of epochs
NUM_EPOCHS = 1

#Recreate an instance of TinyVGG 
model_1 = TinyVGG(input_shape=3, # color_channels RGB 
                  hidden_units=10, 
                  output_shape=len(train_data.classes)).to(device)

# Setup a loss function and optimizer 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(),
                             lr=0.001)


# Start the timer 
start_time = timer()


# Train model_1

model_1_results = train(model=model_1,
                        train_dataloader = train_dataloader_simple,
                        test_dataloader= test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn= loss_fn,
                        epochs = NUM_EPOCHS)

# End teh timer and print out how long it took 
end_time = timer()

print(f" Total training time: {(end_time-start_time):.3f} seconds")

print(model_1_results) 

## 7.8 Plot the loss curves of model 1
print(" A loss curve is a way of tracking you rmodel over time >>>>> ")

# Get the model_1_results keys
print(model_1_results.keys())

# string and a list of floats 
def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary """
    
    # Get the loss values of the results dicitonary (trainnig and test)
    loss = results["train_loss"]
    test_loss= results["test_loss"]
    
    
    # Get accuarcy values of the results dictionary (training and test)
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]
    
    # Figure out how many epochs there were
    epochs = range(len(results["train_loss"]))
    
    # Setup a plot
    plt.figure(figsize=(15, 7))
    
    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend(),
    
    # Plot the accuracy 
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend(),
    plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/visualization/plt_Accuracy_loss_model_2.png")

    
    
print(f"plot the loss and the accurracy ... " * 4)
plot_loss_curves(model_1_results)


### 8. What should an ideal loss curve look like 
print("a loss curve is one of the most helpful ways to troubleshoot a model " * 4)
print(f"Loss curve helps in the following: \n The trend of a loss curve is the loss needs to go down overtime and the accuracy needs to go up overtime . \n **** Key notes**** \n Loss curves: evalulate your model performance over time >>>  \n There of the main differen loss curves in ML: \n 1) Unverfitting >>> the  test loss lower than the train.... \n 2) Overfitting >>> The train loss lower than the test .... \n 3) Just Right fitting >>>> Train and test loss are closer to each other .. \n \n Other ways ot interpreting Loss Curves: \n 1) The model won't train .... \n 2) Model loss exploded .... \n 3) The metrics are contradictory .. \n 4) Testing loss is Too Damn high, ... \n 5) Model Gets Stuck ... \n\n\n 1. Underfitting: When your model loss on the training and test datasets could be lower  on test than train data .... \n 2. Overfitting: when the training loss is lower than the testing loss, because the model is learning the training data too well but fails to generalize well to the test dataset  .. \n 3. Just right curve: We want ideally our training loss to reduce as our test loss with respect to the slightly lower on the training set than test set ... because the model is expose to the raining data and never seen the test data before ..  so it might be a little be lower on the trainig dataset than the test dataset ..... \n\n\n\n So, underfitting the model's loss could be lower, \n Overfitting the model is learning the training data too well... \n Just right fitting: ideally the model train and test loss goes with the similary or close declines  ")

print(f"Overcome overfitting: \n1. more data \n 2. data augmentation \n3. transfer learning \n 4. better data, \n 5. simplify your model. \n 6.use learning rate decay : slowly decrease teh learning rate ... The closer you get to convergence the lower you set the learning rate Like searching the coin at the back of Couch ..   \n 7. early topping : leep track and save the model before the testing loss starts to increase ... .. ")

print(f"Dealing with underfitting: When a loss in't as low as it should to be. our model is not fitting the data very well  \n1. add more layers unints... 2. tweak the laerning rate .. \n 3. traing for longer ...\n.  ML is about balance between over/underfitting \n 4. user transfer learaing \n use less reqularization ... "
      f"\n Underfitting .... increase model probabilities to learn leads to overfitting so be in between ...  ")

        
print(f"TinyVGG with data augmentation ....  "* 3)
# modellling experiment this time with some data augmentation

#9.1 Create transform with data augmentation
train_transform_trivial = transforms.Compose([
                                            transforms.Resize(size=(64, 64)),
                                            transforms.TrivialAugmentWide(num_magnitude_bins=31),# at of 0 -31 images
                                            transforms.ToTensor()
    ])
test_transform_trivial = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

#Create train test dataset/ dataloader with data augmentation
print(f"Turns image folder into datastest ")
train_data_augmented = datasets.ImageFolder(root=train_dir,
                                            transform=train_transform_trivial)
test_data_simple = datasets.ImageFolder(root=test_dir,
                                        transform=test_data_simple)

print(train_data_augmented, test_data_simple)

print(f"Turn datasets into dataloader ...")
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

torch.manual_seed(42)

train_dataloader_augmented = DataLoader(dataset=train_data_augmented,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=NUM_WORKERS)
test_datalaoder_simple = DataLoader(dataset=test_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=NUM_WORKERS)

print(train_dataloader_augmented, test_datalaoder_simple)

print(f"Construct and train model_2. \n with augmented data ")
# Create model_1 and send it to the target device ...
torch.manual_seed(42)
model_2 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(train_data_augmented.classes)).to(device)

# inspect model 2
print(model_2)

print("After a model and dataloader >>>> \n creaet loss function and optimizer \n train() and evalualte the model ")

#set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# set the number of epochs
NUM_EPOCHS = 1

# Set up the loss function/ Criterion for reducing ...
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_2.parameters(),
                             lr=0.001)

# start timer
start_time = timer()

# Train_2_results
model_2_results = train(model = model_2,
                        train_dataloader = train_dataloader_augmented,
                        test_dataloader = test_dataloader_simple,
                        optimizer = optimizer,
                        loss_fn = loss_fn,
                        epochs= NUM_EPOCHS,
                        device=device
                        )

# End the timer and print out how long it took
end_time = timer()

print(f"Total training time for model_2: {end_time-start_time:.3f} seconds")

# Plot the model .. evaluating the performance of the model by plotting the loss curve ... to evaluation the performance over time

plot_loss_curves(model_2_results) # model_2_results


# 10 Compare our model results
print(f"After evaluation experiments: \n we need to evaluate two models ... \n1. Hardcoding "
      f"\n2. PyTorch + Tensorboard "
      f"\3. Weights and Biases ... multiple experiments"
      f"\n 4. start a new run ... "
      f"\n 5. MLFlow .. ")


model_1_df = pd.DataFrame(model_1_results)
model_2_df = pd.DataFrame(model_2_results)
print(model_1_df.head())
print(model_2_df.head())

print("Plot  "*3)
plt.figure(figsize=(15, 10))

# Get numbeer of epochs
epochs = range(len(model_1_df))
print(f"Number of epoch: ... {epochs}")

# plot for a trian loss
plt.subplot(1, 2, 1)
plt.plot(epochs, model_1_df["train_loss"], label="Model 1")
plt.plot(epochs, model_2_df["train_loss"],label="Model 2")
plt.title("Train Loss")
plt.subplot(1, 2, 2)
plt.plot(epochs, model_1_df["train_acc"], label="Model 1")
plt.plot(epochs, model_2_df["train_acc"],label="Model 2")
plt.title("Train Loss")
plt.xlabel("Epochs")
plt.legend()
plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/visualization/Train_loss_acc_MODEL_1_VS_MODEL_2.png")

 # plot for a test loss and accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, model_1_df["test_loss"], label="Model 1")
plt.plot(epochs, model_2_df["test_loss"],label="Model 2")
plt.title("Test loss")
plt.subplot(1, 2, 2)
plt.plot(epochs, model_1_df["test_acc"], label="Model 1")
plt.plot(epochs, model_2_df["test_acc"], label="Model 2")
plt.title("Test Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/visualization/Test_loss_acc_MODEL_1_VS_MODEL_2.png")

print(f"prediction on a custom image ... "* 3)
# We have to make sure our custom image is the same format as the our model trained on...
print(f"1. In tensor form with datatype (torh.float32) "
      f"\n 2. shape 64 * 64 * 3 "
      f"\n on the right device  ")

custom_image_path = "/home/mhamdan/DeepLearning_PyTorch_2022/food.jpg"
#print(custom_image_path, '\n', custom_image_path.shape, custom_image_path.dtype ) #AttributeError: 'str' object has no attribute 'shape'

# read an image in to PyTorch ..
custom_image_unit8 = torchvision.io.read_image(str(custom_image_path))
print(custom_image_unit8, '\n', custom_image_unit8.shape, '\n', custom_image_unit8.dtype)
print(f"*"* 8)
custom_image_unit8 = custom_image_unit8.permute(1, 2, 0)
print(custom_image_unit8.shape)#, custom_image_path.dtype)AttributeError: 'str' object has no attribute 'dtype'
plt.imshow(custom_image_unit8)
plt.title('custom_image')
plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/custom_image_unit8.jpg")
print(f"*"* 8)

print(f"Custom image tensor: \n {custom_image_unit8}")
print(f"Custom image shape: \n {custom_image_unit8.shape}")
print(f"Custom image data type: \n {custom_image_unit8.data.dtype}")

print(f"Make a prediction " * 5)

# 11.2 Making a prediction on a custom image with a trained PyTorch model
# Try t make a prediction on a an image in unit8 format



# Load in the custom image and convert to torch.float32

custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)
# print(f"Custom image tensor custom_image: \n {custom_image}")
# print(f"Custom image shape custom_image: \n {custom_image.shape}")
# print(f"Custom image data type custom_image: \n {custom_image.data.dtype}")
#
#
# custom_image1 = torchvision.io.read_image(str(custom_image_path)).type(torch.float32) / 255.
# print(f"Custom image tensor custom_image: \n {custom_image1}")
# print(f"Custom image shape custom_image: \n {custom_image1.shape}")
# print(f"Custom image data type custom_image: \n {custom_image1.data.dtype}")
# # plt.imshow(custom_image1)#TypeError: Invalid shape (3, 120, 180) for image data
# #
# # plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/custom_image_unit8.jpg")
#
#
# create a traonsform pipeline to resize image

custom_image_transform = transforms.Compose([
                                transforms.Resize(size=(64, 64))
])
#
# # Trannsrom target image
# custom_image_transformed = custom_image_transform(custom_image1)
#
# # shape oringinal and transformed
# print(f"Original image shape: {custom_image1.shape}")
# print(f"Transformed image shape: {custom_image_transformed.shape}")
# #plt.imshow(custom_image_transformed) #TypeError: Invalid shape (3, 64, 64) for image data
#
# #plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/custom_image_transformed.jpg")
# custom_image_transformed_with_batch_size = custom_image_transformed.unsqeeze(dim=0)
# print(f"Transformed image shape squeezed : {custom_image_transformed_with_batch_size.shape}")#AttributeError: 'Tensor' object has no attribute 'unsqeeze'
#
# # This needs now, the right batch size ....
#
# model_2.eval()
# with torch.inference_mode():
#     custom_image_pred_logits = model_2(custom_image_transformed_with_batch_size.to(device))
#
# print(f"Pred logits ********** : \n {custom_image_pred_logits}")
# print(f"Class names ... {class_names}")
#
# print(f"To make a prediction on a custom image do the following:\n "
#       f"1. Load the image and turn it into a tensor \n "
#       f"2. Make sure the image was the same datatype as the model (torch.float32) \n "
#       f"3. Make sure the image was the same shape as the data the model was trained on (3, 64, 64) with a batch size .. (1, 3, 64, 64)"
#       f"\n 4. Make sure the image was on the same device as our model  ")
#
#
#
# print(f"rule out put from our model " * 3)
# # Convert logits -> prediction probabilities
# print(f"Pred logits: \n {custom_image_pred_logits}")
# custom_image_preds_porbs = torch.softmax(custom_image_pred_logits, dim=1) # the dim=1 >>> is the inner bracket of list
# print("Prediction probabilities")
# print(custom_image_preds_porbs)
#
# # Convet prediction probabilities -> prediction labels
# custom_image_pred_labels = torch.argmax(custom_image_preds_porbs, dim=1)
# print("Prediction labels")
# print(custom_image_pred_labels)
# print("Idex our class name with the custom image pred labels")
# print(class_names[custom_image_pred_labels])
#
#
# print(f" Putting custom image prediction together: butlding a funciton ")
#
# print(f"Idal outcome: "
#       f"\n 1. A function where we pass an image path to an have our model predict on that image and plot the image + predction ")
#

# funciton
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str]= None,
                        transform=None,
                        device = device):
    """Makes a prediction on a target image with a trainednd plots the image and predicton ."""
    # Lood in the image
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # Divide the image pixel values by 255 to get them between [0, 1]
    target_image /= 255.

    # Transform if necessary
    if transform:
        target_image = transform(target_image)

    # Make sure the model is on the target device
    model.to(device)

    # Trun on eval/inference mode and make a prediction
    model.eval()

    with torch.inference_mode():
        # Add an extra dimension to the image (this is  the batch dimension)
        # eg. our model will predict on batches of 1x image
        target_image = target_image.unsqueeze(dim=0)

        # make a prediction on the image with an extra dimension (out put  logits)

        target_image_pred = model(target_image.to(device))

        # Convert the logits to prediction probabilities
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

        # Convert prediction probabilities -> prediction labels
        target_image_pred_labels = torch.argmax(target_image_pred_probs, dim=1)

        # Plot the image algonside the prediciton and predicton probabilities
        plt.imshow(target_image.squeeze().permute(1, 2, 0)) # Remove that  dimension and rearrange the sahpeep to be HWC
        if class_names:
            title = f"Pred: {class_names[target_image_pred_labels.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
        else:
            title = f"Pred: {target_image_pred_labels} | Prob: {target_image_pred_probs.max().cpu():.3f}"
        plt.title(title)
        plt.axis('tight')
        plt.savefig("/home/mhamdan/DeepLearning_PyTorch_2022/PLOT_PRED_ALL_TOGETHER.jpg")



print(f"pred_and_plot_image "*5)
pred_and_plot_image(model=model_2,
                    image_path= custom_image_path,
                    class_names=class_names,
                    transform= custom_image_transform,
                    device = device)




print(f"Predicting on custom data ... \n "
      f"1. Data in right datatype -> torch.float32"
      f"\n. 2. Data on same device as a model \n "
      f"3. Data in correct shape and add the batch size by squeeze(dim=0)   ")


