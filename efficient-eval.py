import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models, datasets
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# **Cihaz Seçimi (GPU varsa kullan)**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# **EfficientNet Modelini Yükle**
model = models.efficientnet_b0(pretrained=False)  # Eğitilmiş model dosyanızı kullanacağız
num_classes = 7
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # Son katmanı 7 sınıfa göre değiştir
model = model.to(device)

# **Eğitilmiş modeli yükle**
model.load_state_dict(torch.load("C:/Users/dteme/OneDrive/Masaüstü/skin cancer/efficientnet/best_model_efficientnet.pth", map_location=device))
model.eval()

# **Tek bir görüntü üzerinde test etme fonksiyonu**
def predict_image_with_scores(image_path):
    image = Image.open(image_path).convert("RGB")

    # **Görüntü ön işleme adımları**
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNet için giriş boyutu
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = transform_test(image).unsqueeze(0).to(device)  # Batch formatına getir

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)

    labels = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    predicted_label = labels[predicted.item()]
    print("Ham Skorlar:", output.cpu().numpy())
    print("Olasılıklar:", probabilities.cpu().numpy())
    
    return predicted_label

# **Örnek Kullanım (Tek bir görüntü tahmini)**
image_path = "C:/Users/dteme/OneDrive/Masaüstü/skin cancer/HAM10000/HAM10000_images_full/ISIC_0032319.jpg"
print("Tahmin edilen sınıf:", predict_image_with_scores(image_path))


# **Validation Veri Setini Yükleme**
val_data_dir = "C:/Users/dteme/OneDrive/Masaüstü/skin cancer/data_backup/val"  # Validation veri seti dizininiz
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNet giriş boyutu
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(root=val_data_dir, transform=val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
class_names = val_dataset.classes  # Sınıf isimleri (ImageFolder alfabetik sıraya göre ayarlar)

# **Confusion Matrix Hesaplama**
def evaluate_confusion_matrix(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return cm, all_labels, all_preds

# **Confusion Matrix'i Hesapla ve Görselleştir**
cm, true_labels, predicted_labels = evaluate_confusion_matrix(model, val_loader, device)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("EfficientNet Confusion Matrix")
plt.show()
