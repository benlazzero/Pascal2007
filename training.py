import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastai.vision.all import *
from PIL import Image
from torch import *
from torchvision import models, transforms

torch.cuda.empty_cache()
torch.set_default_device('cuda')
device = torch.device("cuda")
device2 = torch.device("cpu")

# download data/put in dataframe
path = untar_data(URLs.PASCAL_2007)
df = pd.read_csv(path/'train.csv')

# split data
valid_df = df[df['is_valid'] == True]
train_df = df[df['is_valid'] == False]

valid_data = valid_df['fname'].apply(lambda x: path/'train'/x).values.tolist()
valid_targets = valid_df['labels'].apply(lambda x: x.split(' ')).values.tolist()
train_data = train_df['fname'].apply(lambda x: path/'train'/x).values.tolist()
train_targets = train_df['labels'].apply(lambda x: x.split(' ')).values.tolist()

def find_unique(listofarrs):
  flat = []
  for arr in listofarrs:
    for item in arr:
      flat.append(item)
  np_flat = np.array(flat)
  unique = np.unique(np_flat)
  return unique

# resize images -> tensor
transform = transforms.Compose([
  transforms.Resize((280, 280)),
  transforms.ToTensor(),
])

# Identify unique classes and create a mapping from class to index
unique_classes = find_unique(train_targets)
class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

# make training and validation datasets
class MyDataset(torch.utils.data.Dataset):
  def __init__(self, data, targets, transform, class_to_idx):
    self.data = data
    self.targets = targets
    self.transform = transform
    self.class_to_idx = class_to_idx

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    image = Image.open(self.data[idx]).convert('RGB')
    target = self.targets[idx]

    if self.transform:
      image = self.transform(image)
    
    # one hot encoding
    target_onehot = torch.zeros(len(self.class_to_idx))
    for label in target:
      target_onehot[self.class_to_idx[label]] = 1.

    return image, target_onehot

train_set = MyDataset(train_data, train_targets, transform=transform, class_to_idx=class_to_idx)
valid_set = MyDataset(valid_data, valid_targets, transform=transform, class_to_idx=class_to_idx)   

t_loader = DataLoader(train_set, batch_size=64, shuffle=True)
v_loader = DataLoader(valid_set, batch_size=64, shuffle=True)

def trainer(t_load, v_load, num_epochs=7, lr=1e-3, f_epochs=3):
  # get model
  model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
  
  # freeze all layers before replace output
  for param in model.parameters():
    param.requires_grad = False
    
  # replace output layer
  num_features = model.fc.in_features
  model.fc = torch.nn.Linear(num_features, len(class_to_idx))

  # Loss function - Binary cross entropy
  criterion = torch.nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr)
    
  # training frozen
  for f_epoch in arange(f_epochs):
    print('Epoch {}/{}'.format(f_epoch+1, f_epochs))
    for phase in ['train', 'val']:
      if phase == 'train':
        print("loading t")
        dataloader = t_load
        model.train()
      else:
        print("loading v")
        dataloader = v_load
        model.eval()

      f_running_loss = 0.0
      f_running_corrects = 0
      f_total_samples = 0
      
      # iterate through data
      for inputs, labels in dataloader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        
        # forward pass
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          
          # backwards 
          if phase == 'train':
            loss.backward()
            optimizer.step()
            
        # stats
        f_running_loss += loss.item() * inputs.size(0)
        f_total_samples += inputs.size(0)
        f_preds = torch.sigmoid(outputs) >= 0.4
        f_correct_samples = (f_preds == labels).all(dim=1)  
        f_running_corrects += f_correct_samples.sum().item()
        print("current avg loss", f_running_loss / f_total_samples)
        
      f_epoch_loss = f_running_loss / len(dataloader.dataset)
      f_epoch_acc = f_running_corrects / f_total_samples 

    
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, f_epoch_loss, f_epoch_acc))

  # training unfrozen
  epochs=num_epochs
  # unfreeze
  for param in model.parameters():
    param.requires_grad = True

  # new optimizer with params
  optimizer = torch.optim.Adam(model.parameters(), lr)

  for epoch in arange(epochs):
    print('Epoch {}/{}'.format(epoch+1, epochs))
    
    for phase in ['train', 'val']:
      if phase == 'train':
        print("loading t")
        dataloader = t_load
        model.train()
      else:
        print("loading v")
        dataloader = v_load
        model.eval()

      running_loss = 0.0
      running_corrects = 0
      total_samples = 0
      
      # iterate through data
      for inputs, labels in dataloader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        
        # forward pass
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          
          # backwards 
          if phase == 'train':
            loss.backward()
            optimizer.step()
            
        # stats
        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        preds = torch.sigmoid(outputs) >= 0.4
        correct_samples = (preds == labels).all(dim=1)  
        running_corrects += correct_samples.sum().item()
        print("current avg loss", running_loss / total_samples)
        
      epoch_loss = running_loss / len(dataloader.dataset)
      epoch_acc = running_corrects / total_samples 

    
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
  print('Finished Training')
  return model.to('cuda')
  
# invoke trainer
model = trainer(t_loader, v_loader, num_epochs=6, lr=1e-4, f_epochs=3)

# test on test set
dft = pd.read_csv(path/'test.csv')
test_data = dft['fname'].apply(lambda x: path/'test'/x).values.tolist()
test_targets = dft['labels'].apply(lambda x: x.split(' ')).values.tolist()
test_set = MyDataset(test_data, test_targets, transform=transform, class_to_idx=class_to_idx)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

def test_model(model, dataloader):
    print("running testset")
    model.to(device)
    model.eval()  # set model to evaluation mode
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():  # Do not calculate gradients, we are only testing
        for inputs, labels in dataloader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs = model(inputs)  # Run the model on the inputs
            preds = torch.sigmoid(outputs) >= 0.4  # Get the model's predictions
            correct_samples = (preds == labels).all(dim=1)  
            running_corrects += correct_samples.sum().item()
            total_samples += labels.size(0)

        accuracy = running_corrects / total_samples 
        print(f'Accuracy: {accuracy:.4f}')
        
        # Now let's look at some of the correct/incorrect predictions
        for i in arange(10):
            inputs=inputs.to(device2)
            labels=labels.to(device2)
            preds=preds.to(device2)
            input_img = inputs[i]
            true_label = (labels[i] == 1).nonzero().flatten()
            pred_label = (preds[i] == 1).nonzero().flatten()
            t = [idx_to_class[idx.item()] for idx in true_label]
            p = [idx_to_class[idx.item()] for idx in pred_label]

            
            if true_label.equal(pred_label):
                print(f"Image {i} was CORRECTLY classified:", p, "=", t)
            else:
                print(f"Image {i} was INCORRECTLY classified:", p, "!=", t)
            
            # Display the image
            plt.imshow(input_img.permute(1, 2, 0))  # You might need to transform the image back to the original domain (0, 1) or (0, 255)
            plt.show()

test_model(model, test_loader)
