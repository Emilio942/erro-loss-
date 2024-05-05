import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Definition der Testfunktion für Optimierer
def evaluate_optimizers(model_class, parameters, optimizers, train_loader, val_loader, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()  # Annahme: Klassifikationsproblem

    results = {}

    for opt_name, opt_class in optimizers.items():
        model = model_class(**parameters).to(device)  # Modell initialisieren
        optimizer = opt_class(model.parameters())  # Optimierer initialisieren
        best_val_loss = float('inf')

        for epoch in range(epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        results[opt_name] = best_val_loss

    return results

# Beispiel: Verwendung der Funktion mit einem fiktiven Modell und Datensätzen
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv = nn.Conv2d(1, 20, 5)
        self.fc = nn.Linear(20, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Dummy-Daten
train_loader = DataLoader([], batch_size=10)  # Beispiel DataLoader
val_loader = DataLoader([], batch_size=10)

# Optimiererliste
optimizers = {
    'SGD': optim.SGD,
    'Adam': optim.Adam,
    'RMSprop': optim.RMSprop
}

# Funktionsaufruf
model_params = {}
results = evaluate_optimizers(DummyModel, model_params, optimizers, train_loader, val_loader)
print(results)
