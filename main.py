import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import kagglehub

print("Загрузка датасета anime-faces-vs-human-faces...")
path = kagglehub.dataset_download("sanyam1992000/anime-faces-vs-human-faces")
print("Путь к датасету:", path)

data_path = os.path.join(path, "Data")
print(f"Путь к папке Data: {data_path}")


def load_images(data_path, img_size=(64, 64), max_per_class=1000):
    images = []
    labels = []

    print("\nЗагрузка изображений...")

    anime_path = os.path.join(data_path, "anime")
    anime_files = [f for f in os.listdir(anime_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:max_per_class]

    for i, filename in enumerate(anime_files):
        try:
            img_path = os.path.join(anime_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(img_size)
            images.append(np.array(img))
            labels.append(0)
        except:
            continue

    human_path = os.path.join(data_path, "human")
    human_files = [f for f in os.listdir(human_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:max_per_class]

    for i, filename in enumerate(human_files):
        try:
            img_path = os.path.join(human_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(img_size)
            images.append(np.array(img))
            labels.append(1)
        except:
            continue

    print(f"Загружено: {len(images)} изображений")
    print(f"Anime: {labels.count(0)}, Human: {labels.count(1)}")

    return np.array(images), np.array(labels)


print("\nЗагрузка изображений (1000 штук от каждого)...")
train_data, train_labels = load_images(data_path, img_size=(64, 64), max_per_class=1000)

X_train, X_test, y_train, y_test = train_test_split(
    train_data, train_labels,
    test_size=0.2,
    random_state=42,
    stratify=train_labels
)

print(f"\nТренировочные данные: {len(X_train)}")
print(f"Тестовые данные: {len(X_test)}")

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001), input_shape=(64, 64, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same',
           kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),

    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

print("\n" + "=" * 50)
print("НАЧАЛО ОБУЧЕНИЯ")
print("=" * 50)

history = model.fit(
    X_train, y_train_cat,
    epochs=50,
    batch_size=32,
    validation_split=0.15,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nТочность на тестовых данных: {test_acc:.4f} ({test_acc * 100:.1f}%)")

model.save('cnn_anime_human_model.h5')

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Тренировочная точность')
plt.plot(history.history['val_accuracy'], label='Валидационная точность')
plt.title('Точность модели')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Тренировочные потери')
plt.plot(history.history['val_loss'], label='Валидационные потери')
plt.title('Потери модели')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()
