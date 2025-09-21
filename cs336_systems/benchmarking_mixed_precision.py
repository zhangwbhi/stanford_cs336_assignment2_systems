import torch
import torch.nn as nn

# Define the model
class ToyModel(nn.Module):
    def __init__(self, in_features:int, out_features:int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print("After fc1 (ReLU):", x.dtype)
        x = self.ln(x)
        print("After LayerNorm:", x.dtype)
        x = self.fc2(x)
        print("Final logits:", x.dtype)
        return x

# Setup
device = "cuda"
model = ToyModel(5, 3).to(device)
x = torch.randn(2, 5, device=device)
target = torch.randint(0, 3, (2,), device=device)

scaler = torch.amp.GradScaler("cuda")
criterion = nn.CrossEntropyLoss()

# Forward pass under autocast
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    print("Model parameter dtype inside autocast:", model.fc1.weight.dtype)
    logits = model(x)
    loss = criterion(logits, target)
    print("Loss dtype:", loss.dtype)

    # Backward pass
    scaler.scale(loss).backward()

    # Print gradient dtype
    print("Gradient dtype of fc1.weight:", model.fc1.weight.grad.dtype)