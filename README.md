# facenet-face-recognition

This repository contains a demonstration of face recognition using the FaceNet network (https://arxiv.org/pdf/1503.03832.pdf) and a webcam. Our implementation feeds frames from the webcam to the network to determine whether or not the frame contains an individual we recognize.

## How to use

To install all the requirements for the project run

	pip install -r requirements.txt

In the root directory. After the modules have been installed you can run the project by using python

	python facenet.py

## NOTE

We are using the Windows 10 Text-to-Speech library to output our audio message, so if you want to run it on a different OS then you will have to replace the speech library used in facenet.py

## FaceNet — пример простой системы распознавания лиц с открытым кодом Github
(Взято из https://te.legra.ph/FaceNet--primer-prostoj-sistemy-raspoznavaniya-lic-s-otkrytym-kodom-Github-02-01, https://t.me/BookPythonFebruary 01, 2022)

![](https://te.legra.ph/file/0ff202e6267b6466da00d.png)

Apple использует Face ID, OnePlus --- технологию Face Unlock. [Baidu использует распознавание лица вместо ID-карт для обеспечения доступа в офис](https://www.youtube.com/watch?v=wr4rx0Spihs), а при повторном пересечении границы в ОАЭ вам нужно только посмотреть в камеру.

В статье разбираемся, как сделать простейшую сеть распознавания лиц самостоятельно с помощью FaceNet.

[Ссылка на Гитхаб, кому нужен только код](https://github.com/Skuldur/facenet-face-recognition)

### Немного о FaceNet

FaceNet --- нейронная сеть, которая учится преобразовывать изображения лица в компактное евклидово пространство, где дистанция соответствует мере схожести лиц. Проще говоря, чем более похожи лица, тем они ближе.

### Триплет потерь

FaceNet использует особую функцию потерь называемую TripletLoss. Она минимизирует дистанцию между якорем и изображениями, которые содержат похожую внешность, и максимизирует дистанцую между разными.

![](https://cdn-images-1.medium.com/max/1600/1*wBxUThEsuAfDq3s-knIyyQ.png)

-   f(a) это энкодинг якоря
-   f(p) это энкодинг похожих лиц (positive)
-   f(n) это энкодинг непохожих лиц (negative)
-   Альфа --- это константа, которая позволяет быть уверенным, что сеть не будет пытаться оптимизировать напрямую f(a) --- f(p) = f(a) --- f(n) = 0
-   [...]+ экиввалентено max(0, sum)

### Сиамские сети

FaceNet --- сиамская сеть. Сиамская сеть --- тип архитектуры нейросети, который обучается диффиренцированию входных данных. То есть, позволяет научиться понимать какие изображения похожи, а какие нет.

![](https://neurohive.io/wp-content/uploads/2018/11/1z9gzhxpLxqqsXI7r6yzgog-e1542353098866.jpeg)

Сиамские сети состоят из двух идентичных нейронных сетей, каждая из которых имеет одинаковые точные веса. Во-первых, каждая сеть принимает одно из двух входных изображений в качестве входных данных. Затем выходы последних слоев каждой сети отправляются в функцию, которая определяет, содержат ли изображения одинаковые идентификаторы.

В FaceNet это делается путем вычисления расстояния между двумя выходами.

### Реализация

Переходим к практике.

В реализации мы будем использовать [Keras](https://keras.io/) и [Tensorflow](https://www.tensorflow.org/). Кроме того, мы используем два файла утилиты из репозитория [deeplayning.ai](https://github.com/shahariarrabby/deeplearning.ai/tree/master/COURSE%204%20Convolutional%20Neural%20Networks/Week%2004/Face%20Recognition), чтобы абстрагироваться от взаимодействий с сетью FaceNet.

-   fr_utils.py содержит функции для подачи изображений в сеть и получения кодирования изображений;
-   inception_blocks_v2.py содержит функции для подготовки и компиляции сети FaceNet.

### Компиляция сети FaceNet

Первое, что нам нужно сделать, это собрать сеть FaceNet для нашей системы распознавания лиц.

```python
import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
from keras import backend as K
K.set_image_data_format('channels_first')
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
def triplet_loss(y_true, y_pred, alpha = 0.3):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,
               positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,
               negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)
```

Мы начнем инициализпцию нашей сети со входа размерности (3, 96, 96). Это означает, что картинка передается в виде трех каналов RGB и размерности 96×96 пикселей.

Теперь давайте определим Triplet Loss функцию. Функция в сниппете кода выше удовлетворяет уравнению Triplet Loss, которое мы определили в предыдущей секции.

Если вы не знакомы с фреймворком TensorFlow, ознакомьтесь с документацией.

Сразу после того, как мы определили функцию потерь, мы можем скомпилировать нашу систему распознавания лиц с помощью Keras. Мы будем использовать [Adam optimizer](https://keras.io/optimizers/#adam) для минимизации потерь, подсчитанных с помощью функции Triplet Loss.

### Подготовка базы данных

Теперь когда мы скомпилировали FaceNet, нужно подготовить базу данных личностей, которых сеть будет распознавать. Мы будем использовать все изображения, которые лежат в директории images.

***Замечание:**** мы будем использовать по одному изображения на человека в нашей реализации. FaceNet достаточно мощна, чтобы распознать человека по одной фотографии.*

```python
def prepare_database():
    database = {}
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_path_to_encoding(file, FRmodel)
    return database
```

Для каждого изображения мы преобразуем данные изображения в 128 float чисел. Этим занимается функция **img_path_to_encoding. **Функция принимает на вход путь до изображения и «скармливает» изображение нашей распознающей сети, после чего возвращают результаты работы сети.

Как только мы получили закодированное изображения в базе данных, сеть наконец готова приступить к распознаванию!

### Распознавание лиц

Как уже обсуждалось ранее, FaceNet пытается минимизировать расстояние между схожими изображениями и максимизировать между разными. Наша реализация использует данную информацию для того, чтобы определить, кем является человек на новой картинке.

```python
def who_is_it(image, database, model):
    encoding = img_to_encoding(image, model)

    min_dist = 100
    identity = None

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(db_enc - encoding)
        print('distance for %s is %s' %(name, dist))
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.52:
        return None
    else:
        return identity
```

Загружаем новое изображение в функцию img_to_encoding. Функция обрабатывает изображения, используя FaceNet и возвращает закодированное изображение. Теперь мы можем сделать предположение о наиболее вероятной личности этого человека.

Для этого подсчитываем расстояние между полученным новым изображением и каждым человеком в нашей базе данных. Наименьшая дистанция укажет на наиболее вероятную личность человека.

Наконец, мы должны определить действительно ли совпадают личности на картинке и в базе. Следующий кусок кода как раз для этого:

```python
 if min_dist > 0.52:
     return None
 else:
     return identity
```

Магическое число 0.52 получено методом проб и ошибок. Для вас это число может отличатся, в зависимости от реализации и данных. Попробуйте настроить самостоятельно.

На GitHub есть демо работы полученной сети, с входом от простой вебкамеры.

### Заключение

Теперь вы знаете, как работают технологии распознавания лиц и можете сделать собственную упрощенную сеть распознавания, используя предварительно подготовленную версию алгоритма FaceNet на python.

Автор: [Станислав Литвинов](https://neurohive.io/ru/author/st_akrenor/)

Источник: <https://medium.freecodecamp.org/making-your-own-face-recognition-system-29a8e728107c>

Github: <https://github.com/Skuldur/facenet-face-recognition>
