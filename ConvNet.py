import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary




# Hyper-parameters
num_epochs = 10
learning_rate = 0.001

# Image preprocessing modules
# Define the data augmentation transform
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Resize((50,50)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Defining the path to the dataset
train_root = '/home/aimsgh-02/Music/split_data/not_val/train'
test_root = '/home/aimsgh-02/Music/split_data/not_val/test'


# Loading the dataset
train_dataset = torchvision.datasets.ImageFolder(
    root=train_root,
    transform=transform
)

test_dataset = torchvision.datasets.ImageFolder(
    root=test_root,
    transform=transform
)

# Specifying batch size and creating data loaders
batch_size = 100
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)





class MyConvNet(nn.Module):
    def __init__(self, num_class = 2):
        super(MyConvNet, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        for _ in range(32):
            self.cnn_layers.add_module('conv', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
            self.cnn_layers.add_module('relu', nn.ReLU(inplace=True))

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Instantiate the model
model = MyConvNet()
print(model)

# Use torchsummary to display the model summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
summary(model.to(device), (3, 32, 32))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate


import time

# Start the timer
start_time = time.time()

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Test the model
model.eval()


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'convnet.ckpt')



end_time = time.time()  
    
# Calculate the running time
running_time = end_time - start_time

# Print or log the running time
print(f"Running time: {running_time:.2f} seconds")

