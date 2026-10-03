# Multi-Class Image Classification via Transfer Learning

This project demonstrates how to perform **multi-class image classification** using **Transfer Learning** with **PyTorch** and a pre-trained **ResNet18** model.

The model is trained on the **STL-10** dataset, which contains 10 different object classes.

---

# 📌 Project Overview

The goal of this project is to build an image classification model that receives an image and predicts **one class** from a fixed set of possible classes.

For example:

```text
Input Image
     │
     ▼
  ResNet18
     │
     ▼
Class Probabilities
     │
     ▼
   Argmax
     │
     ▼
Predicted Class
```

If the input image contains a dog, the model might produce:

```text
dog       → 92%
cat       → 4%
horse     → 2%
deer      → 1%
...
```

The final prediction would be:

```text
dog
```

This is a **multi-class classification** problem because each image is assigned to one target class.

---

# 🧠 Transfer Learning

This project uses **Transfer Learning** with a pre-trained ResNet18 model.

Instead of training a neural network from scratch, we start with a model that has already learned useful visual features from a large image dataset.

These learned features include:

- Edges
- Textures
- Shapes
- Patterns
- Object parts
- More complex visual representations

We then adapt the final classification layer to the STL-10 dataset.

---

## Why Use Transfer Learning?

Training a deep neural network from scratch can require:

- Large amounts of data
- Significant computational resources
- Long training times

A pre-trained model already contains useful visual representations.

By reusing these features and replacing the final classification layer, we can adapt the model to a new classification task much more efficiently.

---

# 📊 Dataset

This project uses the **STL-10** dataset.

STL-10 contains **10 object classes**:

| Class | Description |
|---|---|
| `airplane` | Airplanes |
| `bird` | Birds |
| `car` | Cars |
| `cat` | Cats |
| `deer` | Deer |
| `dog` | Dogs |
| `frog` | Frogs |
| `horse` | Horses |
| `ship` | Ships |
| `truck` | Trucks |

The model therefore has:

```text
10 output neurons
```

because there are 10 possible classes.

---

# 🛠️ Technologies Used

This project uses:

- **Python**
- **PyTorch**
- **Torchvision**
- **ResNet18**
- **STL-10**
- **Transfer Learning**
- **CUDA / GPU acceleration**
- **Matplotlib**
- **NumPy**

---

# 🚀 Step-by-Step Implementation

## Step 1: Environment Setup & GPU Check

The project can be run using **Google Colab**.

For GPU acceleration in Google Colab:

```text
Runtime
   ↓
Change runtime type
   ↓
T4 GPU
```

Using a GPU can significantly speed up model training.

---

## Import Required Libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
```

---

## Check GPU Availability

```python
# Verify GPU availability
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")
```

If CUDA is available, the output should be similar to:

```text
Using device: cuda
```

Otherwise:

```text
Using device: cpu
```

The `device` variable will later be used to move both the model and tensors to the GPU when available.

---

# Step 2: Data Transformations

ResNet18 expects images in a specific format and size.

We therefore need to preprocess the STL-10 images before passing them to the model.

Typical preprocessing includes:

- Resizing
- Random data augmentation during training
- Converting images to tensors
- Normalizing pixel values

Example:

```python
data_transforms = {

    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ]),

    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
}
```

### Why Normalize the Images?

The ResNet18 model was originally trained on ImageNet.

The following values are commonly used for ImageNet normalization:

```text
Mean:
[0.485, 0.456, 0.406]

Standard deviation:
[0.229, 0.224, 0.225]
```

Using the same normalization helps the input data remain compatible with the pre-trained model's learned representations.

---

# Step 3: Download and Load the STL-10 Dataset

Torchvision provides a built-in implementation of the STL-10 dataset.

```python
train_dataset = datasets.STL10(
    root='./data',
    split='train',
    download=True,
    transform=data_transforms['train']
)

test_dataset = datasets.STL10(
    root='./data',
    split='test',
    download=True,
    transform=data_transforms['test']
)
```

The dataset will automatically be downloaded into:

```text
./data
```

when:

```python
download=True
```

is enabled.

---

# Step 4: Create DataLoaders

We use PyTorch `DataLoader` objects to load the images in batches.

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2
)
```

### DataLoader Configuration

| Parameter | Training | Testing |
|---|---:|---:|
| Batch size | 32 | 32 |
| Shuffle | Yes | No |
| Workers | 2 | 2 |

### Why Shuffle Training Data?

During training, shuffling prevents the model from seeing the training examples in the exact same order every epoch.

This helps reduce unwanted ordering effects during optimization.

For testing, shuffling is unnecessary because we are only evaluating the model.

---

# Step 5: Load the Pre-Trained ResNet18

We now load a ResNet18 model with pre-trained ImageNet weights.

```python
model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)
```

At this point, the model has already learned general-purpose visual features.

---

# Step 6: Freeze the Feature Extractor

The convolutional layers of ResNet18 contain the feature extraction part of the model.

We can freeze these parameters so they are not updated during the initial training phase.

```python
for param in model.parameters():
    param.requires_grad = False
```

This means:

```text
ResNet18 Feature Extractor
        │
        ├── Frozen
        │
        ▼
Learned Image Features
```

We will only train the new classification layer.

---

# Step 7: Replace the Classification Head

The original ResNet18 model was designed for ImageNet, which contains 1000 classes.

STL-10 contains only 10 classes.

Therefore, we need to replace the original fully connected layer.

First, retrieve the number of input features:

```python
num_features = model.fc.in_features
```

Then replace the final layer:

```python
model.fc = nn.Linear(
    num_features,
    10
)
```

The architecture now looks like:

```text
Input Image
     │
     ▼
ResNet18
     │
     ▼
Feature Extractor
     │
     ▼
Fully Connected Layer
     │
     ▼
10 Output Classes
```

---

# Step 8: Move the Model to the Device

We move the model to the GPU if one is available.

```python
model = model.to(device)
```

If the device is:

```text
cuda
```

the model will run on the GPU.

If the device is:

```text
cpu
```

the model will run on the CPU.

---

# Step 9: Define the Loss Function

Because this is a **multi-class classification** problem, we use:

```python
criterion = nn.CrossEntropyLoss()
```

`CrossEntropyLoss` is appropriate when each image belongs to exactly one class.

For example:

```text
Image → Dog
```

rather than:

```text
Image → Dog + Person
```

---

# Step 10: Define the Optimizer

Since the ResNet18 feature extractor is frozen, we only need to optimize the new classification layer.

```python
optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)
```

This means the optimizer updates the parameters of:

```text
model.fc
```

while the frozen ResNet18 layers remain unchanged.

---

# Step 11: Training Loop

We can now train the model.

```python
epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Track loss
        running_loss += loss.item() * inputs.size(0)

        # Calculate predictions
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    epoch_loss = running_loss / len(train_dataset)

    epoch_accuracy = (
        100 * correct / total
    )

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Accuracy: {epoch_accuracy:.2f}%"
    )
```

---

# Step 12: Model Evaluation

After training, we evaluate the model on the test dataset.

```python
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

test_accuracy = (
    100 * correct / total
)

print(
    f"Test Accuracy: {test_accuracy:.2f}%"
)
```

The model's predictions are obtained using:

```python
torch.max(outputs, 1)
```

which returns the class with the highest output score.

---

# 🔍 Understanding the Output

Suppose the model produces:

```text
airplane = 2.1
bird     = 0.4
car      = 8.7
cat      = 1.2
deer     = 0.7
dog      = 0.3
frog     = 0.1
horse    = 0.8
ship     = 0.5
truck    = 0.9
```

The largest value is:

```text
car = 8.7
```

Therefore:

```text
Predicted Class = car
```

The raw outputs from the final layer are called **logits**.

For multi-class classification, the class with the highest logit is selected.

---

# 🧠 Multi-Class Classification Logic

The fundamental prediction process is:

```text
                 Input Image
                      │
                      ▼
                 ResNet18
                      │
                      ▼
                  10 Logits
                      │
                      ▼
              Highest Logit
                      │
                      ▼
              Predicted Class
```

For example:

```text
[1.2, 0.4, 8.7, 0.2, 0.8, ...]
           ▲
           │
        highest
```

The model predicts the corresponding class.

---

# 🔄 Multi-Class vs Multi-Label

It is important to distinguish this project from a multi-label classification problem.

| Feature | Multi-Class | Multi-Label |
|---|---|---|
| Classes per image | One | Multiple possible |
| Target | Single class ID | Multi-hot vector |
| Example | `dog` | `dog + person` |
| Loss | `CrossEntropyLoss` | `BCEWithLogitsLoss` |
| Activation concept | Softmax | Sigmoid |
| Prediction | `argmax()` / highest logit | Threshold |
| Classes compete? | Yes | No |

### Multi-Class Example

```text
Image
  ↓
Dog
```

### Multi-Label Example

```text
Image
  ↓
Dog + Person + Car
```

---

# 🔬 Why ResNet18?

ResNet18 is a relatively lightweight convolutional neural network that provides a good balance between:

- Computational cost
- Training speed
- Model capacity
- Classification performance

Its residual connections help very deep networks learn effectively by improving gradient flow.

The model can therefore be used as a strong baseline for transfer-learning experiments.

---

# 📈 Transfer Learning Workflow

The complete transfer-learning process can be summarized as:

```text
              Pre-trained ResNet18
                       │
                       ▼
             ImageNet Features
                       │
                       ▼
          Freeze Feature Extractor
                       │
                       ▼
           Replace Classification Head
                       │
                       ▼
              10 STL-10 Classes
                       │
                       ▼
                  Train Head
                       │
                       ▼
                Evaluate Model
```

---

# ⚠️ Important Note About Fine-Tuning

The initial implementation freezes the entire ResNet18 model and trains only the final classification layer.

Strictly speaking, this is often called **feature extraction** rather than full fine-tuning.

A later improvement can be to unfreeze some of the deeper ResNet layers:

```python
for param in model.layer4.parameters():
    param.requires_grad = True
```

Then those layers can be adapted to the STL-10 dataset as well.

This creates a more traditional fine-tuning setup:

```text
Early Layers
     │
     └── Frozen

Deep Layers
     │
     └── Trainable

Classification Head
     │
     └── Trainable
```

---

# 🚀 Possible Improvements

The basic implementation can be extended in several ways.

## 1. Fine-Tune Deeper Layers

Instead of freezing the entire backbone, unfreeze the final ResNet blocks.

```python
for param in model.layer4.parameters():
    param.requires_grad = True
```

---

## 2. Use a Learning Rate Scheduler

A learning-rate scheduler can gradually reduce the learning rate during training.

For example:

```python
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)
```

Then call:

```python
scheduler.step()
```

after each epoch.

---

## 3. Add More Data Augmentation

The training pipeline can be expanded with transformations such as:

```python
transforms.RandomResizedCrop(224)
transforms.RandomRotation(10)
transforms.ColorJitter(...)
```

These can help the model become more robust to variations in the input images.

---

## 4. Track Validation Accuracy

Instead of evaluating only at the end, a validation loop can be added after every training epoch.

This makes it possible to monitor:

```text
Training Loss
Validation Loss
Training Accuracy
Validation Accuracy
```

and identify potential overfitting.

---

## 5. Save the Trained Model

The trained weights can be saved using:

```python
torch.save(
    model.state_dict(),
    "resnet18_stl10.pth"
)
```

The model can later be restored using:

```python
model.load_state_dict(
    torch.load("resnet18_stl10.pth")
)
```

---

# 📦 Requirements

Install the required packages with:

```bash
pip install torch torchvision matplotlib numpy
```

If you are using Google Colab, PyTorch, Torchvision, NumPy, and Matplotlib are generally already available.

---

# ▶️ Running the Project

## Option 1: Google Colab

1. Create a new Google Colab notebook.
2. Enable GPU acceleration.
3. Run the import section.
4. Check CUDA availability.
5. Download the STL-10 dataset.
6. Create the DataLoaders.
7. Load the pre-trained ResNet18.
8. Replace the classification head.
9. Train the model.
10. Evaluate it on the test set.

---

## Option 2: Local Python Environment

Install the required dependencies:

```bash
pip install torch torchvision matplotlib numpy
```

Then run the Python script or notebook.

The STL-10 dataset will be downloaded automatically when:

```python
download=True
```

is specified.

---

# 🧩 Complete Architecture

The final model can be represented as:

```text
                    Input Image
                         │
                         ▼
                 Resize → 224×224
                         │
                         ▼
                    Normalize
                         │
                         ▼
                  Pre-trained
                   ResNet18
                         │
                         ▼
                Feature Extractor
                         │
                         ▼
                 Fully Connected
                      Layer
                         │
                         ▼
                  10 Class Logits
                         │
                         ▼
               Highest Probability
                         │
                         ▼
                 Predicted Class
```

---

# 📚 Key Concepts Demonstrated

This project demonstrates:

- Transfer Learning
- Feature Extraction
- Fine-Tuning concepts
- Convolutional Neural Networks
- ResNet18
- STL-10
- Multi-Class Classification
- Cross-Entropy Loss
- Adam Optimizer
- Data Augmentation
- Image Normalization
- PyTorch `Dataset`
- PyTorch `DataLoader`
- GPU acceleration with CUDA
- Model evaluation
- Model saving and loading

---

# 📝 Summary

This project builds a **multi-class image classifier** using a pre-trained **ResNet18** model and the **STL-10** dataset.

The central idea is to reuse visual features learned from ImageNet instead of training a convolutional neural network entirely from scratch.

The workflow is:

```text
STL-10 Dataset
      │
      ▼
Image Preprocessing
      │
      ▼
Pre-trained ResNet18
      │
      ▼
Freeze Feature Extractor
      │
      ▼
Replace Classification Head
      │
      ▼
10 Output Classes
      │
      ▼
CrossEntropyLoss
      │
      ▼
Adam Optimizer
      │
      ▼
Training
      │
      ▼
Evaluation
```

The key distinction from multi-label classification is that **each image is assigned one class**, so `CrossEntropyLoss` and selecting the highest-scoring class are appropriate.

In contrast, a multi-label classifier can assign multiple classes to the same image and typically uses `BCEWithLogitsLoss` with independent sigmoid probabilities.
