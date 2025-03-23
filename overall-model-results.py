import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import os

# Cihaz seçimi (GPU varsa kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Validation veri seti için transformasyon (ResNet/EfficientNet/VGG için 224x224 giriş boyutu)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation veri seti dizini (kendi dizininizi belirtin)
val_data_dir = "C:/Users/dteme/OneDrive/Masaüstü/bionluk/skin cancer/balanced_dataset/val"
val_dataset = datasets.ImageFolder(root=val_data_dir, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
class_names = val_dataset.classes  # Sınıflar alfabetik sıraya göre (örn. ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'])

# Değerlendirme fonksiyonu: Modelin doğrulama seti üzerindeki metriklerini hesaplar.
def evaluate_model(model, data_loader, device, criterion=None):
    model.eval()
    total = 0
    correct = 0
    losses = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += torch.sum(preds == labels).item()
            # Toplanan etiketler ve tahminler
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if criterion is not None:
                loss = criterion(outputs, labels)
                losses.append(loss.item() * images.size(0))
    accuracy = correct / total
    if criterion is not None:
        avg_loss = sum(losses) / total
    else:
        avg_loss = None

    # Ek metrikler: precision, recall, F1 score (weighted)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    if criterion is not None:
        return accuracy, avg_loss, precision, recall, f1
    return accuracy, precision, recall, f1

# Fonksiyonlar: Her modelin ağırlıklarını yükleyen fonksiyonlar.
def load_vgg_model(weights_path):
    model = models.vgg16(pretrained=False)
    num_classes = 7
    model.classifier[6] = nn.Linear(4096, num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def load_resnet_model(weights_path):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    model = model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def load_efficientnet_model(weights_path):
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
    model = model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

# Ağırlık dosyalarının yollarını belirtin (kendi dosya isimlerinize göre güncelleyin)
vgg_weights = "C:/Users/dteme/OneDrive/Masaüstü/bionluk/skin cancer/vgg16/best_model_vgg.pth"
resnet_weights = "C:/Users/dteme/OneDrive/Masaüstü/bionluk/skin cancer/restnet50/final_skin_cancer_model.pth"
efficientnet_weights = "C:/Users/dteme/OneDrive/Masaüstü/bionluk/skin cancer/efficientnet/best_model_efficientnet.pth"

# Her modeli yükle ve değerlendirme metriklerini hesapla.
criterion = nn.CrossEntropyLoss()  # Sınıflandırma için kayıp fonksiyonu

vgg_model = load_vgg_model(vgg_weights)
vgg_acc, vgg_loss, vgg_precision, vgg_recall, vgg_f1 = evaluate_model(vgg_model, val_loader, device, criterion)

resnet_model = load_resnet_model(resnet_weights)
resnet_acc, resnet_loss, resnet_precision, resnet_recall, resnet_f1 = evaluate_model(resnet_model, val_loader, device, criterion)

efficientnet_model = load_efficientnet_model(efficientnet_weights)
efficientnet_acc, efficientnet_loss, efficientnet_precision, efficientnet_recall, efficientnet_f1 = evaluate_model(efficientnet_model, val_loader, device, criterion)

# Sonuçları sözlükte toplayalım
results = {
    "VGG16": {"Accuracy": vgg_acc, "Loss": vgg_loss, "Precision": vgg_precision, "Recall": vgg_recall, "F1 Score": vgg_f1},
    "ResNet50": {"Accuracy": resnet_acc, "Loss": resnet_loss, "Precision": resnet_precision, "Recall": resnet_recall, "F1 Score": resnet_f1},
    "EfficientNetB0": {"Accuracy": efficientnet_acc, "Loss": efficientnet_loss, "Precision": efficientnet_precision, "Recall": efficientnet_recall, "F1 Score": efficientnet_f1}
}

print("Model Sonuçları:")
for model_name, metrics in results.items():
    print(f"{model_name}: Accuracy = {metrics['Accuracy']:.4f}, Loss = {metrics['Loss']:.4f}, "
          f"Precision = {metrics['Precision']:.4f}, Recall = {metrics['Recall']:.4f}, F1 Score = {metrics['F1 Score']:.4f}")

# Sonuçları pandas DataFrame'e dönüştürelim
df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
print(df)
# Bar grafikler ile görselleştirme
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Doğruluk grafiği
sns.barplot(ax=axes[0], data=df, x="Model", y="Accuracy", palette="viridis")
axes[0].set_ylim(0, 1)
axes[0].set_title("Model Accuracy Comparison")
axes[0].set_ylabel("Accuracy")

# Kayıp grafiği
sns.barplot(ax=axes[1], data=df, x="Model", y="Loss", palette="magma")
axes[1].set_title("Model Loss Comparison")
axes[1].set_ylabel("Loss")

plt.tight_layout()
plt.show()

# Her model için sınıf bazlı sonuçları ve confusion matrix'i hesaplayan fonksiyon
def evaluate_model_per_class(model, data_loader, device):
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
    # Confusion matrix
    conf_mat = confusion_matrix(all_labels, all_preds)
    # Sınıf bazlı precision, recall ve f1 score (average=None her sınıf için ayrı sonuç üretir)
    per_class_precision = precision_score(all_labels, all_preds, average=None)
    per_class_recall = recall_score(all_labels, all_preds, average=None)
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    
    # Sınıf bazlı accuracy: Her sınıf için (y = c) doğru tahminlerin o sınıfa ait örnek sayısına oranı
    per_class_accuracy = []
    for i in range(len(class_names)):
        # Sadece gerçek etiketleri i olan tüm örnekler
        indices = [j for j, label in enumerate(all_labels) if label == i]
        if indices:
            correct = np.sum(np.array(all_preds)[indices] == i)
            acc = correct / len(indices)
        else:
            acc = 0.0
        per_class_accuracy.append(acc)
    return per_class_accuracy, per_class_precision, per_class_recall, per_class_f1, conf_mat

# Her model için sınıf bazlı metrikleri hesaplayalım
vgg_class_acc, vgg_class_prec, vgg_class_rec, vgg_class_f1, vgg_conf_mat = evaluate_model_per_class(vgg_model, val_loader, device)
resnet_class_acc, resnet_class_prec, resnet_class_rec, resnet_class_f1, resnet_conf_mat = evaluate_model_per_class(resnet_model, val_loader, device)
efficientnet_class_acc, efficientnet_class_prec, efficientnet_class_rec, efficientnet_class_f1, efficientnet_conf_mat = evaluate_model_per_class(efficientnet_model, val_loader, device)

# Her model için sınıf bazlı sonuçları tablo haline getirelim
vgg_df = pd.DataFrame({
    "Class": class_names,
    "Accuracy": vgg_class_acc,
    "Precision": vgg_class_prec,
    "Recall": vgg_class_rec,
    "F1 Score": vgg_class_f1
})
resnet_df = pd.DataFrame({
    "Class": class_names,
    "Accuracy": resnet_class_acc,
    "Precision": resnet_class_prec,
    "Recall": resnet_class_rec,
    "F1 Score": resnet_class_f1
})
efficientnet_df = pd.DataFrame({
    "Class": class_names,
    "Accuracy": efficientnet_class_acc,
    "Precision": efficientnet_class_prec,
    "Recall": efficientnet_class_rec,
    "F1 Score": efficientnet_class_f1
})

print("VGG16 - Sınıf Bazlı Sonuçlar:")
print(vgg_df)
print("\nResNet50 - Sınıf Bazlı Sonuçlar:")
print(resnet_df)
print("\nEfficientNetB0 - Sınıf Bazlı Sonuçlar:")
print(efficientnet_df)

# Her modelin confusion matrix'lerini görselleştirelim
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sns.heatmap(vgg_conf_mat, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=class_names, yticklabels=class_names)
axes[0].set_title("VGG16 Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(resnet_conf_mat, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=class_names, yticklabels=class_names)
axes[1].set_title("ResNet50 Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

sns.heatmap(efficientnet_conf_mat, annot=True, fmt="d", cmap="Blues", ax=axes[2],
            xticklabels=class_names, yticklabels=class_names)
axes[2].set_title("EfficientNetB0 Confusion Matrix")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.show()
