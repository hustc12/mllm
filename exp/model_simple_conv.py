import os
import torch
import torch.nn as nn

class SimpleConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        return x

# Create model instance
model = SimpleConvNet()

# Create a dummy input to test the model
dummy_input = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32

# Test the model
output = model(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")

# Save the model
dest_path = os.path.join(os.path.dirname(os.getcwd()), 'models')
if (os.path.exists(dest_path) == False):
    os.makedirs(dest_path)

model_path = os.path.join(dest_path, 'simple_conv_model.pth')

torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

print('***************************')
model = torch.load(model_path)
for key in model:
    print(f"{key}  Shape: {model[key].shape}")