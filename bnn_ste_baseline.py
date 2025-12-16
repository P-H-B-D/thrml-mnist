import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os

BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
N_EPOCHS = 50
HIDDEN_UNITS = 64
VISUALIZATION_SAMPLES = 4
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.where(input >= 0, 1.0, -1.0)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

def binarize(x):
    return STEFunction.apply(x)

class FloatWeightLinear(nn.Linear):
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

class BinarizedActivation_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = FloatWeightLinear(784, HIDDEN_UNITS) 
        self.fc2 = FloatWeightLinear(HIDDEN_UNITS, 10)
        self.bn1 = nn.BatchNorm1d(HIDDEN_UNITS)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.bn1(x)
        x = binarize(x) 
        x = self.fc2(x)
        return x

def get_data_loaders():
    ds_train = tfds.load('mnist', split='train', shuffle_files=True, as_supervised=True)
    ds_test = tfds.load('mnist', split='test', shuffle_files=False, as_supervised=True)
    
    ds_train = ds_train.batch(BATCH_SIZE, drop_remainder=True).prefetch(1)
    
    ds_test = ds_test.batch(BATCH_SIZE).prefetch(1)

    def data_generator(ds):
        BINARIZATION_THRESHOLD = 0.5 
        
        for batch_imgs, batch_lbls in tfds.as_numpy(ds):
            imgs = batch_imgs.astype(np.float32) / 255.0
            
            imgs = np.where(imgs >= BINARIZATION_THRESHOLD, 1.0, 0.0).astype(np.float32)
            
            imgs = torch.tensor(imgs)
            lbls = torch.tensor(batch_lbls).long()
            yield imgs, lbls
    
    def get_train_gen():
        return data_generator(ds_train)
    
    def get_test_gen():
        return data_generator(ds_test)
            
    return get_train_gen, get_test_gen

@torch.no_grad()
def validate_and_visualize(model, get_test_gen, device, epoch):
    model.eval()
    
    test_loader_gen = get_test_gen()
    test_data, test_target = next(test_loader_gen)
    test_data, test_target = test_data.to(device), test_target.to(device)
    
    output = model(test_data)
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(test_target.view_as(pred)).sum().item()
    total = test_target.size(0)

    val_acc = 100. * correct / total
    print(f"Acc: {val_acc:.2f}% ({correct}/{total})")
    
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    fig.suptitle(f"Epoch {epoch} | Val. Acc: {val_acc:.2f}%", fontsize=14)
    
    indices = np.random.choice(total, VISUALIZATION_SAMPLES, replace=False)
    
    for i, idx in enumerate(indices):
        ax = axes[i // 2, i % 2]
        
        img = test_data[idx].cpu().numpy().reshape(28, 28)
        
        actual_label = test_target[idx].item()
        predicted_label = pred[idx].item()

        ax.imshow(img, cmap='gray')
        ax.set_title(f"P: {predicted_label} | A: {actual_label}", 
                     color='green' if predicted_label == actual_label else 'red',
                     fontsize=9)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"predictions_float_weights.png")
    plt.close(fig)
    
    model.train()
    
    return val_acc

def plot_metrics(train_losses, val_accs, n_epochs):
    epochs = range(1, n_epochs + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    fig.suptitle("Binarized Activation Network Training Metrics", fontsize=16)

    ax1.plot(epochs, train_losses, label='Training Loss', color='blue')
    ax1.set_title('Training Loss vs. Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (CrossEntropy)')
    ax1.grid(True)
    
    ax2.plot(epochs, val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Validation Accuracy vs. Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("training_curves_float_weights.png")
    print("\nTraining curves saved to training_curves_float_weights.png")
    plt.close(fig)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training Binarized Activation Network on {device}...")

model = BinarizedActivation_MLP().to(device) 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

get_train_gen, get_test_gen = get_data_loaders()

train_losses = []
val_accs = []

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    total_loss = 0
    
    train_loader_gen = get_train_gen()
    for batch_idx, (data, target) in enumerate(train_loader_gen):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / (batch_idx + 1)
    print(f"--- Epoch {epoch} Done | Avg Train Loss: {avg_loss:.4f} ---")
    
    train_losses.append(avg_loss)
    
    current_val_acc = validate_and_visualize(model, get_test_gen, device, epoch)
    val_accs.append(current_val_acc)

print("\nBinarized Activation Network Training Complete.")

plot_metrics(train_losses, val_accs, N_EPOCHS)