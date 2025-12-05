# Вариант 10
Датасет Anime face vs Human face

# Загрузка dataset
```

print("Загрузка датасета anime-faces-vs-human-faces...")
path = kagglehub.dataset_download("sanyam1992000/anime-faces-vs-human-faces")
print("Путь к датасету:", path)

data_path = os.path.join(path, "Data")
print(f"Путь к папке Data: {data_path}")

```
Данные загружаются с сервера. После чего указывается путь к файлам, которые будут использоваться в дальнейшей работе.
Далее необходимо стандартизировать файлы.
```

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

```
В данном фрагменте происходит предобработка файлов. Их размер меняется до 64х64 пикселя, цветовая схема меняется на RGB.
После этого изображение преобразуется в массив, который представляет из себя набор данных о каждом пикселе в изображении: высота, ширина и цвет.
Также после обработки всех изображений одного класса, к какждому из них добавляется метка 1 или 0, 0 - аниме лицо, 1 - человеческое лицо.
```
```
После предобработки, для обучения модели загружаются по 1000 изображений каждого класса:
```

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

```
После загрузки данные разбиваются на тренировочное и тестовые в отношении 80 на 20. Для учлучшения показателей данные преобразуются в float32.
То есть модель будет работать с малыми дробными числами вместо больших целых. Например цветовая палитра будет иметь диапазон от 0.0 до 1.0 вместо 0 - 255.
Также применяем One-hot encoding - представление меток в виде бинарного вектора. Такой подход позволяет избежать восприятие меток нейросетью как упорядоченных данных.
Далее построим модель сверточной нейросети. Сверточная нейросеть - нейросеть, специализированная на обработке данных с сетчатой структурой, например изображения.
```

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

```
Используем модель Sequential - модель с последовательным добавлением слоев, где каждый слой имеет один вход и выход.
Модель состоит из 6 последовательных блоков:
# 1 блок:
Сверточный слой с 32 фильтрами, с функцией активации ReLU. Выход по размерам приравнивается к входу, путем добавления нулей по краям.
Используется L2 регуляризация во избежание переобучения модели. Выходные данные форматируются: 64х64 пикселя и 3 цветовых канала.
Batch Normalization - стабилизирует данные, делая сеть менее зависимой от начальных весов, а также позволяет использовать высокие скорости обучения.
Max Pooling - выбирает максимальное значение в каждом окне размерностью 2х2, таким образом сохраняя важные признаки, при этом уменьшая размерность в два раза.
DropOut - отключает случайные 25% нейронов на каждой итерации обучения, что позволяет бороться с переобучением модели.
# 2 блок:
Аналогичен первому, но имеет 64 фильтра.
# 3 блок:
Аналогичен второму, но имеет 128 фильтров.
# 4 блок:
Функция Flatten преобразует наше изображение в одномерный массив состоящий из 8192 значений.
Функция Dense связывает каждый нейрон со всеми входами, формируя полносвязный слой из 128 нейронов, с функцией активации ReLU и L2 регуляризацией.
На каждой итерации обучения отключаются 50% нейронов для борьбы с переобучением. Отключаем больше нейронов, так как полносвязные слой более склонны к переобучению.
# 5 блок:
Аналогичен 4, но уменьшено вдвое количество нейронов, а также случайно отключаются 30% нейронов, а не 50%.
# 6 блок:
Выходной слой, состоящий из двух нейронов, которые подают на выход вероятность того, что изображено аниме лицо или человеческое. Сумма вероятностей равна единице.
Функция активации softmax преобразует вектор значений в распределение вероятностей.

Далее модель компилируется с алгоритмом оптимизации adam - адаптивная скорость обучения.
Функция потерь categorical_crossentropy используется для сравнения результатов предсказания с истинными результатами.
Качество модели характеризуется ее точностью.

функция summary выводит в консоль структуру модели.

Функция early stopping отвечает за раннюю остановку обучения модели, если ее результаты перестают улучшаться на тестовых данных.
Прекращение улучшения оценивается по показаниям потерь для последних 10 эпох обучения. Если модель перестает улучшаться, то возвращается лучший результат.

Функция reduce_lr ускоряет или замедляет обучение модели в зависимости от показателя потерь на последних 5 эпохах обучения. Если качество уменьшается, то скорость замедляется,
если улучшается, то ускоряется.
```

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

```

Выше представленно обучение модели. процесс обучения состоит из 50 эпох по 1360 изображений на каждую эпоху.
В одной эпохе 43 итерации обучения по 32 изображения. Каждый раз изображения перемешиваются, что улучшает качество обучения модели.
```

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
```

Последним шагом становится вывод статистики обучения, которая дает представление о том, как модель обучалась.

```
