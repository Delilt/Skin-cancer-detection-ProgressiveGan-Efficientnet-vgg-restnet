import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
import os
from torch.utils.data import DataLoader
from PIL import Image

# GPU Kullanımı Kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {device}")

# Veri Seti Konumu
data_dir = "C:/Users/dteme/OneDrive/Masaüstü/skin cancer/balanced_dataset"

# Veri Transformasyonu
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Model Tanımlama
def create_vgg16():
    model = models.vgg16(pretrained=True)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 7)  # 7 sınıf var
    return model.to(device)

# Modeli Eğitme Fonksiyonu
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15):
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        print(f"\n📌 Epoch {epoch+1}/{num_epochs}")
        
        # Eğitim Modu
        model.train()
        train_loss, correct_train = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data)

        train_acc = correct_train.double() / len(train_loader.dataset)

        # Doğrulama Modu
        model.eval()
        val_loss, correct_val = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data)

        val_acc = correct_val.double() / len(val_loader.dataset)

        print(f"✅ Eğitim Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"🎯 Doğrulama Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # En iyi modeli kaydet
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model_vgg.pth")
            print("🔥 En iyi model kaydedildi!")

# **Ana Çalışma Bloğu (Windows İçin Gerekli!)**
if __name__ == "__main__":
    # Veri Kümesini Yükleme
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # Modeli, Kayıp Fonksiyonunu ve Optimizasyonu Ayarla
    model = create_vgg16()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Modeli Eğit
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15)
