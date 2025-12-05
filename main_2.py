import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# Загрузка сохраненной модели
model = load_model('cnn_anime_human_model.h5')


# Функция для предобработки изображения
def preprocess_image(image_path, img_size=(64, 64)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img


# Функция для отображения результата
def display_prediction(image, prediction, classes=['Anime', 'Human']):
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Входное изображение')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    bars = plt.bar(classes, prediction[0], color=['blue', 'orange'])
    plt.ylim([0, 1])
    plt.title('Вероятности предсказания')
    plt.ylabel('Вероятность')

    # Добавляем значения на столбцы
    for bar, val in zip(bars, prediction[0]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


# Основная часть программы
def main():
    print("=" * 60)
    print("КЛАССИФИКАТОР: ANIME FACE vs HUMAN FACE")
    print("=" * 60)
    print("0 - Anime Face")
    print("1 - Human Face")
    print("=" * 60)

    while True:
        print("\nВыберите действие:")
        print("1. Проверить на одном изображении")
        print("2. Протестировать на примерах из датасета (если доступно)")
        print("3. Выход")

        choice = input("Введите номер (1/2/3): ")

        if choice == '1':
            image_path = input("Введите путь к изображению: ")

            try:
                # Загрузка и предобработка
                img_array, original_img = preprocess_image(image_path)

                # Предсказание
                prediction = model.predict(img_array, verbose=0)
                anime_prob = prediction[0][0]
                human_prob = prediction[0][1]

                print("\n" + "=" * 40)
                print("РЕЗУЛЬТАТ ПРЕДСКАЗАНИЯ:")
                print("=" * 40)
                print(f"Вероятность Anime: {anime_prob:.4f}")
                print(f"Вероятность Human: {human_prob:.4f}")
                print("-" * 40)

                if anime_prob > human_prob:
                    confidence = max(anime_prob, human_prob)
                    print(f"Предсказание: ANIME FACE")
                    print(f"Уверенность: {confidence:.2%}")
                else:
                    confidence = max(anime_prob, human_prob)
                    print(f"Предсказание: HUMAN FACE")
                    print(f"Уверенность: {confidence:.2%}")

                # Отображаем результат
                display_prediction(original_img, prediction)

            except Exception as e:
                print(f"Ошибка: {e}")
                print("Убедитесь, что путь правильный и файл существует.")

        elif choice == '2':
            print("\nТестирование на примерах из датасета...")
            try:
                # Попробуем найти тестовые изображения
                test_images = []

                # Ищем примеры anime и human
                import glob

                # Поиск anime примеров
                anime_examples = glob.glob('**/*anime*.jpg', recursive=True)[:2] + \
                                 glob.glob('**/*anime*.png', recursive=True)[:2] + \
                                 glob.glob('**/*cartoon*.jpg', recursive=True)[:2]

                # Поиск human примеров
                human_examples = glob.glob('**/*human*.jpg', recursive=True)[:2] + \
                                 glob.glob('**/*human*.png', recursive=True)[:2] + \
                                 glob.glob('**/*real*.jpg', recursive=True)[:2]

                all_examples = list(set(anime_examples + human_examples))

                if len(all_examples) > 0:
                    for img_path in all_examples[:4]:  # Тестируем первые 4
                        try:
                            img_array, original_img = preprocess_image(img_path)
                            prediction = model.predict(img_array, verbose=0)

                            print(f"\nИзображение: {os.path.basename(img_path)}")
                            print(f"Anime: {prediction[0][0]:.3f}, Human: {prediction[0][1]:.3f}")

                            if 'anime' in img_path.lower() or 'cartoon' in img_path.lower():
                                expected = "Anime"
                            else:
                                expected = "Human"

                            predicted = "Anime" if prediction[0][0] > prediction[0][1] else "Human"
                            status = "✓" if expected == predicted else "✗"
                            print(f"Ожидалось: {expected}, Получено: {predicted} {status}")
                        except:
                            continue
                else:
                    print("Не найдены примеры для тестирования.")

            except Exception as e:
                print(f"Ошибка при тестировании: {e}")

        elif choice == '3':
            print("Выход из программы.")
            break

        else:
            print("Некорректный выбор. Попробуйте снова.")


if __name__ == '__main__':
    main()