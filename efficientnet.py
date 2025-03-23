import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# **Cihaz Seçimi**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# **Veri Ön İşleme**
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# **Veri Setlerini Yükle**
data_dir = "C:/Users/dteme/OneDrive/Masaüstü/skin cancer/balanced_dataset"
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform["train"])
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform["val"])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# **EfficientNet Modelini Yükle**
model = models.efficientnet_b0(pretrained=True)  # EfficientNet'in küçük bir versiyonu (daha büyükleri de var)
num_ftrs = model.classifier[1].in_features  # Son katmanı al
model.classifier[1] = nn.Linear(num_ftrs, 7)  # 7 sınıf için çıktıyı ayarla

model = model.to(device)

# **Kayıp Fonksiyonu ve Optimizasyon**
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# **Model Eğitme Fonksiyonu**
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # **Eğitim**
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

        train_loss /= total
        train_acc = correct.double() / total

        # **Doğrulama**
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                total += labels.size(0)

        val_loss /= total
        val_acc = correct.double() / total

        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # **En iyi modeli kaydet**
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model_efficientnet.pth")
            print("En iyi model kaydedildi!\n")

    print(f"En iyi doğruluk: {best_acc:.4f}")

# **Modeli Eğit**
if __name__ == '__main__':
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)

    # **Eğitilmiş Modeli Kaydet**
    torch.save(model.state_dict(), "final_skin_cancer_model_efficientnet.pth")
    print("EfficientNet modeli kaydedildi!")
