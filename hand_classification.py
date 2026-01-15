"""
Модуль обучения классификатора жестов ASL с использованием ConvNeXt Tiny.
Включает загрузку данных, подготовку модели, обучение и оценку.
"""

import torch
import torch.utils.data as data
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import torchvision.transforms.v2 as tfs
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import models
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image


# ============================================================================
# КЛАСС ДЛЯ ТРАНСФОРМАЦИИ ДАННЫХ
# ============================================================================

class ToTransform(data.Dataset):
    """
    Класс для применения трансформаций к разделенному датасету.
    Решает проблему применения разных трансформеров к train/val/test после split.
    """

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        x, y = self.dataset[item]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)


# ============================================================================
# НАСТРОЙКА УСТРОЙСТВА И ТРАНСФОРМАЦИЙ
# ============================================================================

# Определение устройства для вычислений (GPU/CPU)
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Трансформации для обучающей выборки (с аугментацией)
transform_train = tfs.Compose([
    tfs.Resize((224, 224)),  # Приведение к единому размеру
    tfs.ToImage(),  # Конвертация в тензор
    tfs.RandomRotation(15),  # Случайный поворот ±15 градусов
    tfs.ColorJitter(brightness=0.2, contrast=0.2),  # Изменение яркости и контраста
    tfs.ToDtype(torch.float32, scale=True),  # Конвертация в float32
    tfs.Normalize([0.485, 0.456, 0.406],  # Нормализация по ImageNet
                  [0.229, 0.224, 0.225])
])

# Трансформации для валидационной и тестовой выборок (без аугментации)
transform_val_test = tfs.Compose([
    tfs.ToImage(),
    tfs.Resize((224, 224)),
    tfs.ToDtype(torch.float32, scale=True),
    tfs.Normalize([0.485, 0.456, 0.406],
                  [0.229, 0.224, 0.225])
])

# ============================================================================
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================================================

# Загрузка обучающих данных с трансформациями
train = ImageFolder('ASL_cropped_yolo/Train', transform=transform_train)

# Загрузка тестовых данных для последующего разделения
full_dataset = ImageFolder('ASL_cropped_yolo/Test')

# Параметры разделения датасета
total_len = len(full_dataset)  # Общее количество изображений
val_len = int(0.5 * total_len)  # 50% для валидации
test_len = total_len - val_len  # Остальное для тестирования

# Разделение датасета на валидационную и тестовую части
val_data, test_data = data.random_split(full_dataset, [val_len, test_len])

# Применение трансформаций к разделенным датасетам
val_dataset = ToTransform(val_data, transform_val_test)
test_dataset = ToTransform(test_data, transform_val_test)

# Вывод статистики датасета
print(f"Классов: {len(train.classes)}")
print(f"Тренировочных изображений: {len(train)}")
print(f"Валидационных изображений: {val_len}")
print(f"Тестовых изображений: {test_len}")

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ МОДЕЛИ
# ============================================================================

# Загрузка предобученной модели ConvNeXt Tiny
model = models.convnext_tiny(weights='DEFAULT')

# Заморозка всех слоев для transfer learning
for param in model.parameters():
    param.requires_grad = False

# Замена последнего слоя классификатора под нашу задачу
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, len(train.classes))

# Разморозка только последнего слоя для обучения
for param in model.classifier[-1].parameters():
    param.requires_grad = True

# Альтернативный вариант (Swin Transformer, закомментирован)
# model = models.swin_v2_s(weights='DEFAULT')
# for param in model.parameters():
#     param.requires_grad = False
# model.head = nn.Linear(model.head.in_features, len(train.classes))
# model.head.requires_grad = True


# ============================================================================
# СОЗДАНИЕ DATA LOADERS
# ============================================================================

train_loader = data.DataLoader(train, batch_size=32, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)


# ============================================================================
# ФУНКЦИЯ ОБУЧЕНИЯ МОДЕЛИ
# ============================================================================

def train_model(model, train_data, val_data, epochs=None, checkpoint=None):
    """
    Обучение модели классификатора жестов ASL.

    Параметры:
        model: torch.nn.Module - модель для обучения
        train_data: DataLoader - загрузчик обучающих данных
        val_data: DataLoader - загрузчик валидационных данных
        epochs: int - количество эпох обучения
        checkpoint: str - путь к чекпоинту для продолжения обучения

    Возвращает:
        model: обученная модель
        loss_lst: история потерь на тренировке
        loss_lst_val: история потерь на валидации
    """

    # Перемещение модели на выбранное устройство
    model = model.to(device)

    # Настройка оптимизатора (AdamW с регуляризацией)
    optimizer = optim.AdamW(params=model.parameters(),
                            lr=0.0001,
                            weight_decay=0.001)

    # Функция потерь (кросс-энтропия для многоклассовой классификации)
    loss_func = nn.CrossEntropyLoss()

    # Инициализация переменных для отслеживания процесса обучения
    loss_lst = []  # История потерь на тренировке
    loss_lst_val = []  # История потерь на валидации
    start_epoch = 0  # Начальная эпоха
    best_val_loss = 1e10  # Лучшая потеря на валидации

    # Загрузка чекпоинта если указан и существует
    if checkpoint and os.path.exists(checkpoint):
        checkpoint_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint_dict['model_state_dict'])
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        start_epoch = checkpoint_dict['epoch']
        loss_lst = checkpoint_dict['train_loss_history']
        loss_lst_val = checkpoint_dict['val_loss_history']
        print(f"Загружен чекпоинт из эпохи {start_epoch}")

    # Основной цикл обучения
    for epoch in range(start_epoch, start_epoch + epochs):

        # Переключение модели в режим обучения
        model.train()
        loss_mean = 0  # Скользящее среднее потерь
        lm_count = 0  # Счетчик для расчета среднего

        # Прогресс-бар для эпохи обучения
        train_tqdm = tqdm(train_data, leave=True)

        # Итерация по обучающим батчам
        for x_train, y_train in train_tqdm:
            # Перемещение данных на устройство
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            # Прямой проход
            predict = model(x_train)
            loss = loss_func(predict, y_train)

            # Обратный проход и оптимизация
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Обновление скользящего среднего потерь
            lm_count += 1
            loss_mean = (1 / lm_count * loss.item() +
                         (1 - 1 / lm_count) * loss_mean)

            # Обновление прогресс-бара
            train_tqdm.set_description(
                f'Epoch {epoch + 1}/{start_epoch + epochs}, '
                f'loss_mean: {loss_mean:.3f}'
            )

        # =====================================================
        # ВАЛИДАЦИЯ ПОСЛЕ ЭПОХИ
        # =====================================================
        model.eval()  # Переключение в режим оценки
        Q_val = 0  # Суммарные потери на валидации
        val_count = 0  # Счетчик валидационных примеров

        with torch.no_grad():  # Отключение вычисления градиентов
            for x_val, y_val in val_data:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                predict_val = model(x_val)
                loss_q = loss_func(predict_val, y_val)

                Q_val += loss_q.item()
                val_count += 1

        # Расчет средних потерь на валидации
        Q_val /= val_count

        # Сохранение истории обучения
        loss_lst.append(loss_mean)
        loss_lst_val.append(Q_val)

        # Сохранение чекпоинта каждой эпохи
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': loss_lst,
            'val_loss_history': loss_lst_val,
            'best_val_loss': best_val_loss,
        }, f'hand_model_on_{epoch + 1}_epoch_CNN_new.tar')

        # Обновление лучшего результата если текущий лучше
        if Q_val < best_val_loss:
            best_val_loss = Q_val

    # =====================================================
    # ВИЗУАЛИЗАЦИЯ ИСТОРИИ ОБУЧЕНИЯ
    # =====================================================
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_lst) + 1), loss_lst,
             label='Training Loss', marker='o')
    plt.plot(range(1, len(loss_lst_val) + 1), loss_lst_val,
             label='Validation Loss', marker='s')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training History (Total Epochs: {len(loss_lst)})')
    plt.xticks(range(1, len(loss_lst) + 1))
    plt.legend()
    plt.show()

    print(f"Финальные потери на обучении: {loss_lst[-1]:.4f}")
    print(f"Финальные потери на валидации: {loss_lst_val[-1]:.4f}")

    return model, loss_lst, loss_lst_val


# ============================================================================
# ФУНКЦИЯ ОЦЕНКИ ТОЧНОСТИ МОДЕЛИ
# ============================================================================

def calculate_test_accuracy(model, test_data, checkpoint_path=None):
    """
    Расчет точности (accuracy) модели на тестовом наборе данных.

    Параметры:
        model: torch.nn.Module - модель для тестирования
        test_data: DataLoader - загрузчик тестовых данных
        checkpoint_path: str - путь к чекпоинту для загрузки весов

    Возвращает:
        accuracy: float - точность модели на тестовом наборе
    """

    # Загрузка весов из чекпоинта если указан
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Загружена модель из {checkpoint_path}")

    # Перемещение модели на устройство и режим оценки
    model.to(device)
    model.eval()

    # Инициализация счетчиков
    correct = 0
    total = 0

    # Оценка на тестовом наборе
    with torch.no_grad():
        for x_test, y_test in test_data:
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            predict_test = model(x_test)
            _, predicted = torch.max(predict_test, 1)

            correct += (predicted == y_test).sum().item()
            total += y_test.size(0)

    # Расчет итоговой точности
    accuracy = correct / total
    print(f"Точность на тестовом наборе: {accuracy:.4f} ({correct}/{total})")

    return accuracy

