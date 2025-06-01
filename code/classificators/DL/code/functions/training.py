# Bibliotecas estándar
import csv  # Para escribir archivos CSV
import os
import argparse
import pickle
from collections import Counter

# Procesamiento de datos y matemáticas
import numpy as np
import pandas as pd

# Machine Learning y evaluación
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from imblearn.under_sampling import RandomUnderSampler
from scipy.sparse import csr_matrix, hstack
import seaborn as sns

# Procesamiento de lenguaje natural
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Visualización
import matplotlib.pyplot as plt

# Weights & Biases (experiment tracking)
import wandb

# models
from model_arquitectres.models import MLP2, HybridAttentionMLP, HybridNeuralStacking, NeuronaModel, NeuronaModelT, HybridNeuralStackingT, BaggingEnsemble

# dataset
class CHDataset(Dataset):
    def __init__(self, X_sparse, y, indices=None):
        self.X = X_sparse
        self.y = y
        self.indices = indices if indices is not None else np.arange(len(y))
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx].toarray().squeeze())
        y = torch.FloatTensor([self.y[idx]])
        index = self.indices[idx]
        return x, y, index


def evaluate_model(model, loader, criterion, device, return_probs=False, att = False, txt_shape = 96):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_indices = []

    with torch.no_grad():
        for inputs, labels, indices in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            print('evaluate att',att)
            if att:
                input_txt, input_cat = inputs[:, :txt_shape], inputs[:,txt_shape:]
                outputs = model(input_txt, input_cat)
            else:
                outputs = model(inputs)

            loss = criterion(outputs, labels)

            probs = outputs.squeeze()
            predicted = (probs > 0.5).float()

            running_loss += loss.item() * inputs.size(0)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_indices.extend(indices.cpu().numpy())

            if return_probs:
                all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total

    if return_probs:
        return epoch_loss, epoch_acc,  np.array(all_preds), np.array(all_labels), np.array(all_indices), np.array(all_probs)
    else:
        return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels), np.array(all_indices)



def evaluate_and_save_best_model_outputs(best_model,test_loader , y_test, balanced_data, original_data, class_names, target_col, name_model, att = False, txt_shape = 96):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.BCELoss()
    print('test_loader len',y_test.shape)
    print('evl1')
    _, _, y_pred_labels, y_true_labels, sample_indices, y_probs = evaluate_model(
        best_model, test_loader, criterion, device, return_probs=True, att = att, txt_shape = txt_shape)

    # 1. Calcular predicciones incorrectas
    print('pred le',len(y_pred_labels))
    print(f"Predicciones incorrectas: {np.sum(y_pred_labels != y_true_labels)}")
    incorrect_mask = (y_pred_labels != y_true_labels)[0]
    incorrect_indices = sample_indices[incorrect_mask]

    print('incorrect mask ',len(incorrect_mask))

    # # 2. Índices originales en original_data
    # incorrect_original_indices = balanced_data.iloc[incorrect_indices]['original_index'].values

    # # 3. Etiquetas reales y predichas correspondientes
    # incorrect_real = y_true_labels[incorrect_mask]
    # incorrect_pred = y_pred_labels[incorrect_mask]

    # print('incorrect pred',len(incorrect_pred))

    # # 4. Crear tabla para wandb
    # incorrect_table_data = [
    #     [int(orig_idx), int(pred), int(real)]
    #     for orig_idx, pred, real in zip(incorrect_original_indices, incorrect_pred, incorrect_real)
    # ]

    # 5. Guardar solo si Real != Pred
    # print(len(incorrect_table_data))

    # # 5. Log a wandb
    # wandb.log({
    #     "Incorrect_Predictions_Details": wandb.Table(
    #         data=incorrect_table_data,
    #         columns=["original_index", "predicted", "true"]
    #     )
    # })

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true_labels, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Best Model')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    wandb.log({"Best_Model_ROC_Curve": wandb.Image(plt)})
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plot_confusion_matrix(cm, 'Best Model Confusion Matrix', class_names)
    wandb.log({"Best_Model_Confusion_Matrix": wandb.Image(plt)})

    # Guardar el modelo
    model_path = f'models/classifiers/{name_model}_bm.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'input_size': best_model[0].in_features if isinstance(best_model, nn.Sequential) else None
    }, model_path)
    wandb.save(model_path)

    print("Evaluación y guardado del mejor modelo completado.")

    return model_path

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def train_epoch(model, train_loader, criterion, optimizer,device, att = False, txt_shape = 96):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Convertir el DataLoader en una lista y hacer shuffle manualmente
    shuffled_loader = list(train_loader)
    np.random.shuffle(shuffled_loader)

    for inputs, labels, _ in shuffled_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        if att:
            input_txt, input_cat = inputs[:,:txt_shape], inputs[:,txt_shape:]
            outputs = model(input_txt, input_cat)
        else:
            outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def single_training_loop(X, y, args, balanced_data, original_data, nepoch, att = False, txt_shape = 96):
    class_names = ['False', 'True']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

    # Count per balance classes
    class_counts = Counter(y_train)
    neg = class_counts[0]
    pos = class_counts[1]
    pos_weight = torch.tensor([neg / pos], dtype=torch.float)

    # Create Dataset
    train_dataset = CHDataset(X_train, y_train)
    test_dataset = CHDataset(X_test, y_test)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if att:
        print('shape cat:', X.shape[1]- txt_shape)
        model = HybridAttentionMLP(X.shape[1]- txt_shape, txt_shape, dropout = 0.8).to(device)
    else:
        model = MLP2(X.shape[1], dropout = 0.8).to(device)

    # Init weights
    model.apply(init_weights)

    criterion = nn.BCELoss()  # output sin promedio para aplicar weights manuales
    optimizer = optim.Adam(model.parameters())
    best_val_loss = float('inf')

    # Definir el scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # Training loop
    for epoch in range(nepoch):
        # 1. train epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, att)

        # 2. test epoch
        print('evl2')
        test_loss_epoch, test_acc_epoch, _, _, _ = evaluate_model(model, test_loader, criterion, device, att = att, txt_shape=txt_shape)
    
        # # 3. Ajustar el learning rate usando el scheduler basado en el val_loss
        # scheduler.step(test_loss_epoch)

        wandb.log({
            f"epoch": epoch,
            f"train_loss": train_loss,
            f"test_loss": test_loss_epoch,
            f"train_acc": train_acc,
            f"test_acc": test_acc_epoch
        })

    # Evaluate on test set
    print('evl3')
    test_loss, test_acc, y_test_pred, y_test_true, _ = evaluate_model(model, test_loader, criterion, device, att = att, txt_shape=txt_shape)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test_true, y_test_pred),
        'precision_macro': precision_score(y_test_true, y_test_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test_true, y_test_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    }

    # Store metrics
    for name, value in metrics.items():
        wandb.log({f"{name}": value})

    # Recall por clase
    for class_idx, class_name in enumerate(class_names):
        recall = recall_score(y_test_true == class_idx, y_test_pred == class_idx, zero_division=0)
        wandb.log({f"recall_{class_name}": recall})
        metrics[f'recall_{class_name}'] = recall

    # Confusion matrix
    cm_test = confusion_matrix(y_test_true, y_test_pred)
    plot_confusion_matrix(cm_test, 'Test Confusion Matrix', class_names)

    print(f"\nClassification Report")
    print(classification_report(y_test_true, y_test_pred, target_names=class_names, digits=4))

    print(len(y_test_true))

    return metrics, model, test_loader, y_test

### for autorcross
from scipy.sparse import vstack
def authorcross_training_loop(X_rpa, y_rpa, X_human, y_human, args, nepoch, att=False, txt_shape=96, datatype_test = None):
    class_names = ['False', 'True']

    if datatype_test == '2rpa':
        X_train_rpa, X_test_rpa, y_train_rpa, y_test_rpa = train_test_split(X_rpa, y_rpa, test_size=0.35, random_state=42)

        X_train = vstack([X_train_rpa, X_human])
        y_train = np.concatenate([y_train_rpa, y_human])
        X_test = X_test_rpa
        y_test = y_test_rpa
    if datatype_test == '2human':
        X_train_human, X_test_human, y_train_human, y_test_human = train_test_split(X_human, y_human, test_size=0.35, random_state=42)

        X_train = vstack([X_train_human, X_rpa])
        y_train = np.concatenate([y_train_human, y_rpa])
        X_test = X_test_human
        y_test = y_test_human

    if datatype_test == 'human':
        # Partición definida explícitamente por autor
        X_train, y_train = X_rpa, y_rpa
        X_test, y_test = X_human, y_human

    if datatype_test == 'rpa':
        # Partición definida explícitamente por autor
        X_train, y_train = X_human, y_human
        X_test, y_test = X_rpa, y_rpa

    # Conteo de clases para pesos
    class_counts = Counter(y_train)
    neg = class_counts[0]
    pos = class_counts[1]
    pos_weight = torch.tensor([neg / pos], dtype=torch.float)

    # Crear datasets y dataloaders
    train_dataset = CHDataset(X_train, y_train)
    test_dataset = CHDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if att:
        print('Attention Activated...')
        print('Shape cat:', X_train.shape[1] - txt_shape)
        model = HybridAttentionMLP(X_train.shape[1] - txt_shape, txt_shape, dropout=0.8).to(device)
    else:
        model = MLP2(X_train.shape[1], dropout=0.8).to(device)

    model.apply(init_weights)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    for epoch in range(nepoch):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, att)
        test_loss_epoch, test_acc_epoch, _, _, _ = evaluate_model(model, test_loader, criterion, device, att=att, txt_shape=txt_shape)

        # scheduler.step(test_loss_epoch)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss_epoch,
            "train_acc": train_acc,
            "test_acc": test_acc_epoch
        })

    print("Evaluación final en test externo (otro autor)")
    test_loss, test_acc, y_test_pred, y_test_true, _ = evaluate_model(model, test_loader, criterion, device, att=att, txt_shape=txt_shape)

    metrics = {
        'accuracy': accuracy_score(y_test_true, y_test_pred),
        'precision_macro': precision_score(y_test_true, y_test_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test_true, y_test_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    }

    for name, value in metrics.items():
        wandb.log({name: value})

    for class_idx, class_name in enumerate(class_names):
        recall = recall_score(y_test_true == class_idx, y_test_pred == class_idx, zero_division=0)
        wandb.log({f"recall_{class_name}": recall})
        metrics[f'recall_{class_name}'] = recall

    cm_test = confusion_matrix(y_test_true, y_test_pred)
    plot_confusion_matrix(cm_test, 'Test Confusion Matrix (Cross Author)', class_names)

    print("Classification Report (Cross Author):")
    print(classification_report(y_test_true, y_test_pred, target_names=class_names, digits=4))

    return metrics, model, test_loader, y_test

# train bagging
def bagging_training_loop(X, y, args, balanced_data, original_data, nepoch, 
                         n_estimators=10, sample_ratio=0.8, att=False, txt_shape=96):
    """
    Entrena un ensamblaje de modelos usando bagging.
    
    Args:
        X: Datos de entrada
        y: Etiquetas
        args: Argumentos de configuración
        balanced_data: Datos balanceados
        original_data: Datos originales
        nepoch: Número de épocas por modelo
        n_estimators: Número de modelos en el ensamblaje
        sample_ratio: Proporción de muestras para cada bootstrap
        att: Usar atención o no
        txt_shape: Tamaño de las características de texto
        
    Returns:
        metrics: Métricas del ensamblaje
        ensemble: Objeto BaggingEnsemble
        test_loader: DataLoader de prueba
        y_test: Etiquetas de prueba
    """
    class_names = ['False', 'True']
    
    # Dividir en train/test (usamos el mismo test para todos los modelos)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Crear DataLoader de test (común para todos)
    test_dataset = CHDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Inicializar el ensamblaje
    ensemble = BaggingEnsemble(
        n_estimators=n_estimators,
        input_size=X.shape[1],
        txt_shape=txt_shape,
        att=att,
        device=device
    )
    
    # Entrenar cada modelo del ensamblaje
    for i in range(n_estimators):
        print(f"\nTraining estimator {i+1}/{n_estimators}")
        
        # 1. Muestreo bootstrap
        n_samples = int(sample_ratio * len(y_train))
        indices = np.random.choice(len(y_train), size=n_samples, replace=True)
        X_bootstrap = X_train[indices]
        y_bootstrap = y_train[indices]
        
        # 2. Crear DataLoader para este bootstrap
        train_dataset = CHDataset(X_bootstrap, y_bootstrap)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        # 3. Entrenar el modelo
        model = ensemble.estimators[i]
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        best_val_loss = float('inf')
        
        for epoch in range(nepoch):
            # Entrenamiento
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, att, txt_shape
            )
            
            # Validación (usamos el conjunto de test común)
            test_loss, test_acc, _, _, _ = evaluate_model(
                model, test_loader, criterion, device, att=att, txt_shape=txt_shape
            )
            
            # Loggear métricas
            wandb.log({
                f"estimator_{i}_epoch": epoch,
                f"estimator_{i}_train_loss": train_loss,
                f"estimator_{i}_test_loss": test_loss,
                f"estimator_{i}_train_acc": train_acc,
                f"estimator_{i}_test_acc": test_acc
            })
    
    # Evaluar el ensamblaje completo
    test_loss, test_acc, y_test_pred, y_test_true, _ = evaluate_model(
        ensemble, test_loader, criterion, device, att=False, txt_shape=txt_shape
    )
    
    # Calcular métricas del ensamblaje
    metrics = {
        'accuracy': accuracy_score(y_test_true, y_test_pred),
        'precision_macro': precision_score(y_test_true, y_test_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test_true, y_test_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    }
    
    # Recall por clase
    for class_idx, class_name in enumerate(class_names):
        recall = recall_score(y_test_true == class_idx, y_test_pred == class_idx, zero_division=0)
        metrics[f'recall_{class_name}'] = recall
    
    # Matriz de confusión
    cm_test = confusion_matrix(y_test_true, y_test_pred)
    plot_confusion_matrix(cm_test, 'Ensemble Test Confusion Matrix', class_names)
    
    print(f"\nEnsemble Classification Report")
    print(classification_report(y_test_true, y_test_pred, target_names=class_names, digits=4))
    
    return metrics, ensemble, test_loader, y_test


### plt conf matrix en wandb
def plot_confusion_matrix(cm, title, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    wandb.log({title: wandb.Image(plt)})
    plt.close()

    return plt.gcf()  # Return the figure object for logging


### save metrics
def save_single_metrics(metrics, class_names, name_model):

    # Extraer métricas generales
    summary_metrics = {
        'Accuracy': metrics['accuracy'],
        'Precision (Macro)': metrics['precision_macro'],
        'Recall (Macro)': metrics['recall_macro'],
        'F1-score (Macro)': metrics['f1_macro'],
        'Recall (True)': metrics['recall_True'],
        'Recall (False)': metrics['recall_False'],
    }

    if 'rpa_recall' in metrics and 'human_recall' in metrics:
        summary_metrics['RPA Recall'] = metrics['rpa_recall']
        summary_metrics['Human Recall'] = metrics['human_recall']

    # Log a W&B
    wandb.log(summary_metrics)

    # Guardar CSV
    csv_path = f'/fhome/amir/TFG/code/classificators/DL/results/{name_model}_single_metrics.csv'
    headers = ['Metric', 'Value']
    rows = [[metric, f"{value:.4f}"] for metric, value in summary_metrics.items()]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)

    # Log W&B Table
    df_metrics = pd.DataFrame(rows, columns=headers)
    wandb.log({f"{name_model}_metrics_table": wandb.Table(dataframe=df_metrics)})


    return csv_path

### HYBRID TRAINING

# dataset multiinput
from torch.utils.data import Dataset
import torch
from scipy.sparse import issparse

class CHDatasetMultiInput(Dataset):
    def __init__(self, macroX, y, indices):

        self.X = {}
        for k, v in macroX.items():
            if issparse(v):
                v = v[indices].toarray()  # Convertimos a denso sólo la parte necesaria
            else:
                v = v[indices]
            self.X[k] = v.astype('float32')  # Aseguramos tipo correcto para PyTorch
        self.y = y[indices].astype('float32')  # Asegura dtype compatible con BCELoss

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        inputs = {k: torch.tensor(v[idx], dtype=torch.float32) for k, v in self.X.items()}
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return inputs, label


def evaluate_model_multiinput(model, loader, criterion, device, return_probs=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_indices = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                inputs_dict, labels, indices = batch
            else:
                # Por compatibilidad si el dataset no entrega índices
                inputs_dict, labels = batch
                indices = torch.arange(len(labels))  # dummy indices

            # Mover a device
            for k in inputs_dict:
                inputs_dict[k] = inputs_dict[k].to(device)
            labels = labels.to(device)

            outputs = model(inputs_dict)
            loss = criterion(outputs, labels)

            probs = outputs.squeeze()
            predicted = (probs > 0.5).float()

            running_loss += loss.item() * labels.size(0)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_indices.extend(indices.cpu().numpy())

            if return_probs:
                all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total

    if return_probs:
        return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels), np.array(all_indices), np.array(all_probs)
    else:
        return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels), np.array(all_indices)


# evaluate model multiinput
def evaluate_and_save_best_model_outputs_multiinput(best_model, test_loader, y_test,
                                                     balanced_data, original_data,
                                                     class_names, target_col, name_model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.BCEWithLogitsLoss()
    _, _, y_pred_labels, y_true_labels, sample_indices, y_probs = evaluate_model_multiinput(
        best_model, test_loader, criterion, device, return_probs=True
    )

    # 1. Calcular predicciones incorrectas
    incorrect_mask = (y_pred_labels != y_true_labels)
    incorrect_indices = sample_indices[incorrect_mask]

    # 2. Índices originales en original_data
    incorrect_original_indices = balanced_data.iloc[incorrect_indices]['original_index'].values

    # 3. Etiquetas reales y predichas correspondientes
    incorrect_real = y_true_labels[incorrect_mask]
    incorrect_pred = y_pred_labels[incorrect_mask]

    # 4. Crear tabla para wandb
    incorrect_table_data = [
        [int(orig_idx), int(pred), int(real)]
        for orig_idx, pred, real in zip(incorrect_original_indices, incorrect_pred, incorrect_real)
    ]

    # 5. Log a wandb
    wandb.log({
        "Incorrect_Predictions_Details": wandb.Table(
            data=incorrect_table_data,
            columns=["original_index", "predicted", "true"]
        )
    })

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true_labels, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Best Model')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    wandb.log({"Best_Model_ROC_Curve": wandb.Image(plt)})
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plot_confusion_matrix(cm, 'Best Model Confusion Matrix', class_names)
    wandb.log({"Best_Model_Confusion_Matrix": wandb.Image(plt)})

    # Guardar el modelo
    model_path = f'models/hybridation_neurons/{name_model}.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        'model_state_dict': best_model.state_dict()
    }, model_path)
    wandb.save(model_path)

    print("Evaluación y guardado del mejor modelo completado.")
    return model_path


# train loop
def hybrid_training_loop_multiinput(macroX, y, models_loaded, macro_txtshape, nepoch=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = ['False', 'True']

    # Split por índice
    train_idx, test_idx = train_test_split(range(len(y)), test_size=0.25, random_state=42)
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Crear DataLoader para cada entrada
    train_dataset = CHDatasetMultiInput(macroX, y, train_idx)
    test_dataset = CHDatasetMultiInput(macroX, y, test_idx)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Calcular pesos para clases desbalanceadas
    class_counts = Counter(y_train)
    pos_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float32).to(device)

    # Crear modelo híbrido con múltiples entradas
    neuron_map = {}
    for k, model_list in models_loaded.items():
        # Cargamos los modelos en modo evaluación
        neurons = []
        for model in model_list:
            model.eval()
            neurons.append(model)
        neuron_map[k] = neurons

    hybrid_model = HybridNeuralStacking(neuron_map, macro_txtshape).to(device)

    # Optimización
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(hybrid_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # Entrenamiento
    for epoch in range(nepoch):
        hybrid_model.train()
        total_loss, total_correct, total = 0, 0, 0

        for batch_x_dict, batch_y in train_loader:
            print('batch_x_dict',batch_x_dict)
            print('batch_y',batch_y)
            for k in batch_x_dict:
                batch_x_dict[k] = batch_x_dict[k].to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = hybrid_model(batch_x_dict).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            preds = (outputs > 0.5).float()
            total_loss += loss.item() * batch_y.size(0)
            total_correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        train_loss = total_loss / total
        train_acc = total_correct / total

        # Evaluación
        test_loss, test_acc, y_test_pred, y_test_true, _ = evaluate_model_multiinput(hybrid_model, test_loader, criterion, device)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })

    # Métricas finales
    metrics = {
        'accuracy': accuracy_score(y_test_true, y_test_pred),
        'precision_macro': precision_score(y_test_true, y_test_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test_true, y_test_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    }

    for name, value in metrics.items():
        wandb.log({f"{name}": value})

    for class_idx, class_name in enumerate(class_names):
        recall = recall_score((y_test_true == class_idx), (y_test_pred == class_idx), zero_division=0)
        wandb.log({f"recall_{class_name}": recall})
        metrics[f'recall_{class_name}'] = recall

    cm_test = confusion_matrix(y_test_true, y_test_pred)
    plot_confusion_matrix(cm_test, 'Test Confusion Matrix', class_names)

    print("\nClassification Report")
    print(classification_report(y_test_true, y_test_pred, target_names=class_names, digits=4))

    return metrics, hybrid_model, test_loader, y_test

## all same time training loopç
def joint_training_loop_multiinput(macroX, y, model_factory_fn, macro_txtshape, nepoch=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = ['False', 'True']

    # Split
    train_idx, test_idx = train_test_split(range(len(y)), test_size=0.25, random_state=42)
    y_train = y[train_idx]
    y_test = y[test_idx]

    train_dataset = CHDatasetMultiInput(macroX, y, train_idx)
    test_dataset = CHDatasetMultiInput(macroX, y, test_idx)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    n_models = 2  # Número de modelos por tipo

    # Crear neuronas desde cero
    neuron_map = {}
    for k in macroX.keys():
        neurons = []
        for _ in range(n_models):  # Número de modelos por tipo
            input_dim = macroX[k].shape[1] - macro_txtshape[k]
            text_dim = macro_txtshape[k]
            base_model = model_factory_fn(input_dim, text_dim)
            neurons.append(NeuronaModelT(base_model))
        neuron_map[k] = neurons

    hybrid_model = HybridNeuralStackingT(neuron_map, macro_txtshape).to(device)

    optimizer = torch.optim.AdamW(hybrid_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    criterion_individual = nn.BCEWithLogitsLoss(reduction='mean')

    for epoch in range(nepoch):
        hybrid_model.train()
        total_loss, total, correct = 0, 0, 0

        for batch_x_dict, batch_y in train_loader:
            for k in batch_x_dict:
                batch_x_dict[k] = batch_x_dict[k].to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            final_output, individual_outputs = hybrid_model(batch_x_dict, return_all=True)

            loss_final = criterion(final_output, batch_y)
            individual_losses = [
                criterion_individual(ind_out, batch_y) for ind_out in individual_outputs.T
            ]
            # wand log
            loss_individual = torch.stack(individual_losses).mean()

            total_loss_batch = loss_final + loss_individual
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item() * batch_y.size(0)
            preds = (final_output > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        test_loss, test_acc, y_test_pred, y_test_true, _ = evaluate_model_multiinput(
            hybrid_model, test_loader, criterion, device
        )

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })

    # Calcular métricas finales
    metrics = {
        'accuracy': accuracy_score(y_test_true, y_test_pred),
        'precision_macro': precision_score(y_test_true, y_test_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test_true, y_test_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
    }

    for name, value in metrics.items():
        wandb.log({f"{name}": value})

    for class_idx, class_name in enumerate(class_names):
        recall = recall_score((y_test_true == class_idx), (y_test_pred == class_idx), zero_division=0)
        wandb.log({f"recall_{class_name}": recall})
        metrics[f'recall_{class_name}'] = recall

    cm_test = confusion_matrix(y_test_true, y_test_pred)
    plot_confusion_matrix(cm_test, 'Test Confusion Matrix', class_names)

    print("\nClassification Report")
    print(classification_report(y_test_true, y_test_pred, target_names=class_names, digits=4))

    return hybrid_model, test_loader, y_test, metrics
