import torch
import torch.nn as nn

class SimpleConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3):
        super(SimpleConvNet, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        # self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        # x = self.relu(x)
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
torch.save(model.state_dict(), 'simple_conv_model.pth')
print("Model saved as 'simple_conv_model.pth'")

print('***************************')
model_path = "simple_conv_model.pth"
model = torch.load(model_path)
for key in model:
    print(f"{key}  Shape: {model[key].shape}")