
```markdown
# Multi-Class Image Classification via Transfer Learning

This project demonstrates fine-tuning a pre-trained **ResNet18** model using **PyTorch** for multi-class image classification on the **STL-10** dataset.

---

## Overview & Approach

We leverage **Transfer Learning** using a pre-trained ResNet18 backbone.

### Why Transfer Learning?
Pre-trained models have already learned feature representations (edges, textures, complex shapes) from millions of images on ImageNet. By freezing early feature extraction layers and replacing the final classification head, we achieve high accuracy (90%+) within minutes rather than hours of training from scratch.

### Dataset
This walkthrough uses the **STL-10** dataset, which consists of 10 target classes:
`airplane`, `bird`, `car`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

---

## Step-by-Step Implementation

### Step 1: Environment Setup & GPU Check

Open a fresh Google Colab notebook and enable GPU acceleration (`Runtime` > `Change runtime type` > `T4 GPU`).

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Verify GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

```

> **Rationale:** Matrix operations required for deep learning execute significantly faster on GPU compute units compared to CPUs.

---

### Step 2: Data Preprocessing & Augmentation

Input images vary in layout, brightness, and positioning. We apply transforms to normalize pixel intensity and data augmentation to mitigate overfitting.

```python
# Data transforms for training (with augmentation) and validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean values
                             [0.229, 0.224, 0.225])   # ImageNet std values
    ]),
    'val': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
}

```

* **Resize:** Neural network architectures expect uniform tensor shapes.
* **RandomHorizontalFlip / RandomRotation:** Introduces artificial variation, teaching the network orientation invariance.
* **Normalize:** Centers pixel intensities around zero with unit variance, stabilizing numerical gradients during backpropagation.

---

### Step 3: Load Dataset & Create DataLoaders

Download the STL-10 dataset and wrap instances inside PyTorch DataLoaders to manage batching and shuffling.

```python
# Download STL-10 dataset
train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=data_transforms['train'])
val_dataset = datasets.STL10(root='./data', split='test', download=True, transform=data_transforms['val'])

# DataLoaders for batch processing
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

classes = train_dataset.classes
print(f"Classes ({len(classes)}): {classes}")

```

> **Rationale:** Passing single samples to the GPU creates execution bottlenecks, while feeding the full dataset causes memory allocation errors. `DataLoader` batches samples (size 32) and shuffles training data every epoch.

---

### Step 4: Model Architecture (Transfer Learning)

Instantiate a pre-trained ResNet18 model, freeze its feature extraction parameters, and swap out the fully connected layer to output logits for 10 classes.

```python
# Load pre-trained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze lower convolutional layers
for param in model.parameters():
    param.requires_grad = False

# Replace output classification layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(classes))

# Transfer model to GPU
model = model.to(device)

```

* **`param.requires_grad = False`:** Freezes network parameters so backpropagation updates only the new output layer, minimizing compute requirements.
* **`model.fc` replacement:** Adjusts the default ImageNet 1000-class output head down to our target subset of 10 categories.

---

### Step 5: Loss Function & Optimizer Setup

```python
# CrossEntropyLoss handles LogSoftmax + NLLLoss internally
criterion = nn.CrossEntropyLoss()

# Optimize only parameters of the newly added FC layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

```

* **CrossEntropyLoss:** Evaluation metric for single-label, multi-class classification tasks.
* **Adam:** Adaptive learning rate optimizer providing fast gradient convergence.

---

### Step 6: Training & Validation Loop

Iterate across training epochs to execute forward passes, compute loss metrics, backpropagate gradients, and track validation performance.

```python
epochs = 5

for epoch in range(epochs):
    # --- TRAINING PHASE ---
    model.train()
    running_loss, running_corrects = 0.0, 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()            # Clear gradients
        outputs = model(inputs)          # Forward pass
        loss = criterion(outputs, labels)# Compute loss
        loss.backward()                  # Backward pass
        optimizer.step()                 # Update weights
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    train_loss = running_loss / len(train_dataset)
    train_acc = running_corrects.double() / len(train_dataset)
    
    # --- VALIDATION PHASE ---
    model.eval()
    val_loss, val_corrects = 0.0, 0
    
    with torch.no_grad(): # Disable gradient calculation for evaluation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            
    val_loss = val_loss / len(val_dataset)
    val_acc = val_corrects.double() / len(val_dataset)
    
    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

```

* **`optimizer.zero_grad()`:** PyTorch accumulates gradients across iterations; explicit resetting prevents stale gradient updates.
* **`torch.no_grad()`:** Disables tracking computation graphs during evaluation, cutting memory overhead and accelerating execution.

---

### Step 7: Inference & Visualization

Evaluate the trained model on a sample image from the validation dataset and display output predictions.

```python
model.eval()

# Retrieve a single evaluation batch
dataiter = iter(val_loader)
images, labels = next(dataiter)

# Select first image sample
img, label = images[0], labels[0]

# Run model inference
with torch.no_grad():
    output = model(img.unsqueeze(0).to(device))
    prob = torch.nn.functional.softmax(output, dim=1)
    pred_class_idx = torch.argmax(prob).item()

# Denormalize image tensor for visualization
img_display = img.numpy().transpose((1, 2, 0))
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_display = std * img_display + mean
img_display = np.clip(img_display, 0, 1)

plt.imshow(img_display)
plt.title(f"True: {classes[label]} | Predicted: {classes[pred_class_idx]} ({prob[0][pred_class_idx]*100:.1f}%)")
plt.axis("off")
plt.show()

```

```

```
