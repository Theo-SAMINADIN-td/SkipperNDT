import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

class PipelineClassifier(nn.Module):
    def __init__(self):
        super(PipelineClassifier, self).__init__()
        self.resnet = models.resnet18(weights=None)
        
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        return self.sigmoid(x)

# 2. Paramètres d'entraînement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PipelineClassifier().to(device)
criterion = nn.BCELoss() # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Exemple de forme de données (Batch de 8 images, 4 canaux, 224x224)
# Note: Les TIF devront être chargés via une librairie comme rasterio ou PIL
dummy_input = torch.randn(8, 4, 224, 224).to(device)
labels = torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0]]).to(device)

# 4. Passage Forward simple
outputs = model(dummy_input)
loss = criterion(outputs, labels)

print(f"Probabilités : {outputs.detach().cpu().numpy().flatten()}")
print(f"Perte : {loss.item()}")