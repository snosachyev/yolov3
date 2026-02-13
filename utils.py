import cv2
import glob
import numpy as np
import tensorflow as tf
import os
import random

from matplotlib import pyplot as plt

from cProfile import label

import settings

from PIL import Image, ImageDraw, ImageFont


# функции для загрузки данных
def parse_label(label_file, img_shape):
    """Парсим YOLO label txt в [x_min, y_min, x_max, y_max, class_id]"""
    boxes = []
    with open(label_file.numpy().decode(), "r") as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.strip().split())
            x_min = (x - w / 2) * img_shape[1]
            y_min = (y - h / 2) * img_shape[0]
            x_max = (x + w / 2) * img_shape[1]
            y_max = (y + h / 2) * img_shape[0]
            boxes.append([x_min, y_min, x_max, y_max, int(cls)])
    return tf.convert_to_tensor(boxes, dtype=tf.float32)


def load_data(img_path):
    """Загружаем картинку + парсим label"""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (settings.IMG_SIZE, settings.IMG_SIZE)) / 255.0

    # label path (YOLO формат)
    label_path = tf.strings.regex_replace(img_path, "images", "labels")
    label_path = tf.strings.regex_replace(label_path, ".jpg", ".txt")

    boxes = tf.py_function(
        func=lambda x: parse_label(x, (settings.IMG_SIZE, settings.IMG_SIZE)),
        inp=[label_path],
        Tout=tf.float32,
    )
    boxes.set_shape([None, 5])  # (num_boxes, [xmin,ymin,xmax,ymax,cls])
    return img, boxes


def make_dataset(img_dir, batch=settings.BATCH_SIZE, shuffle=True):
    files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")]
    ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(files))
    ds = ds.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.padded_batch(batch, padded_shapes=([settings.IMG_SIZE, settings.IMG_SIZE, 3], [None, 5]))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# Функция преобразования предсказаных значений в рамки (yolo_boxes)
def yolo_boxes(pred, anchors, classes):
    # На входе pred размера (S, S, 3, (1+4+80))

    grid_size = tf.shape(pred)[1] # S ячеек в сетке

    # В box_xy и box_wh помещаем сразу по 2 переменные (tx, ty) и (tw, th)
    box_xy, box_wh, score, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1) # раскладываем предсказанную карту по переменным

    # Применяем сигмоидные функции
    box_xy = tf.sigmoid(box_xy)
    score = tf.sigmoid(score)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)


    # Построим сетку S x S
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)

    # Привяжем box_xy к ячейкам сетки, учтем смещения (и снова нормируем к диапазону 0, 1)
    # Фактически мы вычисляем центральное положение рамки относительно размеров сетки (якорного поля)
    b_xy = (box_xy + tf.cast(grid, tf.float32)) /  tf.cast(grid_size, tf.float32) # вычисляем b_xy: (bx, by)

    b_wh = tf.exp(box_wh) * anchors # вычисляем b_wh: (bw, bh), ширина и высота рамки bbox

    box_x1y1 = b_xy - b_wh / 2
    box_x2y2 = b_xy + b_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1) # задаем рамку bbox, как 2 координаты углов


    return bbox, score, class_probs, pred_box


# Подавление не максимумов NMS
def nonMaximumSuppression(outputs, anchors, masks, classes):
    boxes, conf, out_type = [], [], []

    # Блок преобразования трех выходов разных масштабов
    for output in outputs:
        boxes.append(tf.reshape(output[0], (tf.shape(output[0])[0], -1, tf.shape(output[0])[-1])))
        conf.append(tf.reshape(output[1], (tf.shape(output[1])[0], -1, tf.shape(output[1])[-1])))
        out_type.append(tf.reshape(output[2], (tf.shape(output[2])[0], -1, tf.shape(output[2])[-1])))

    # Конкатенируем три масштаба в один
    bbox = tf.concat(boxes, axis=1)
    confidence = tf.concat(conf, axis=1)
    class_probs = tf.concat(out_type, axis=1)

    scores = confidence * class_probs # Оценки считаем как произведение оценок объектности на вероятности классов

    # Применяем NMS из пакета tensorflow (работаем с документацией, смотрим параметры самостоятельно: https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression)
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=settings.YOLO_IOU_THRESHOLD,
        score_threshold=settings.YOLO_SCORE_THRESHOLD)

    return boxes, scores, classes, valid_detections

# Функция вычисления IoU
# Функция interval_overlap вычисляет длину пересечения двух отрезков на одной оси (X или Y).
def interval_overlap(interval_1, interval_2):
    x1, x2 = interval_1 # координаты начала и конца первого отрезка
    x3, x4 = interval_2 # координаты начала и конца второго отрезка
    if x3 < x1:
        return 0 if x4 < x1 else (min(x2,x4) - x1)
    else:
        return 0 if x2 < x3 else (min(x2,x4) - x3)


def intersectionOverUnion(box1, box2):

    # box format: [xmin, ymin, xmax, ymax]

    box1 = tf.expand_dims(box1, -2)
    box2 = tf.expand_dims(box2, 0)

    intersect_mins = tf.maximum(box1[..., :2], box2[..., :2])
    intersect_maxes = tf.minimum(box1[..., 2:], box2[..., 2:])

    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    box1_area = (
        (box1[..., 2] - box1[..., 0]) *
        (box1[..., 3] - box1[..., 1])
    )

    box2_area = (
        (box2[..., 2] - box2[..., 0]) *
        (box2[..., 3] - box2[..., 1])
    )

    union_area = box1_area + box2_area - intersect_area

    return intersect_area / (union_area + 1e-10)



# Преобразование ограничивающих рамок
@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs, classes):

    N = tf.shape(y_true)[0]

    y_true_out = tf.zeros(
      (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, classes):
    outputs = []
    grid_size = 13

    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                    (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
    tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        outputs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs, classes))
        grid_size *= 2

    return tuple(outputs) # [x, y, w, h, obj, class]


def preprocess_image(x_train, size):
    return (tf.image.resize(x_train, (size, size))) / 255

def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
                 (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
                 (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def preprocess_image(x_train, size):
    return (tf.image.resize(x_train, (size, size))) / 255


def polygon_to_bbox(values):
    cls = int(values[0])
    coords = np.array(values[1:]).reshape(-1, 2)
    xmin = coords[:, 0].min()
    xmax = coords[:, 0].max()
    ymin = coords[:, 1].min()
    ymax = coords[:, 1].max()
    return [xmin, ymin, xmax, ymax, cls]


def preprocess_fn(x, y):
    return (preprocess_image(x, settings.SIZE), y)


def target_fn(x, y):
    return (
        x,
        transform_targets(
            y,
            settings.YOLO_ANCHORS,
            settings.YOLO_ANCHOR_MASKS,
            settings.NUM_CLASSES
        )
    )


def build_dataset(img_dir, label_dir):

    train_ds = (
        load_yolo_dataset(
            img_dir,
            label_dir,
            img_size=settings.SIZE,
            max_boxes=20
        )
        .shuffle(512)
        .map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)  # preprocess ДО batch
        #.cache()  # если хватает RAM
        .batch(settings.BATCH_SIZE)
        .map(target_fn, num_parallel_calls=tf.data.AUTOTUNE)  # targets ПОСЛЕ batch
        .prefetch(settings.PREFETCH)
    )

    return train_ds


def load_yolo_dataset(img_dir, label_dir, img_size=416, max_boxes=20):

    img_paths = []
    label_paths = []

    for img_file in os.listdir(img_dir):
        if not img_file.endswith((".jpg", ".jpeg", ".png")):
            continue

        img_paths.append(os.path.join(img_dir, img_file))
        label_paths.append(
            os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
        )

    ds = tf.data.Dataset.from_tensor_slices((img_paths, label_paths))

    def parse_fn(img_path, label_path):

        # IMAGE
        img_raw = tf.io.read_file(img_path)
        img = tf.image.decode_image(img_raw, channels=3)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.float32) / 255.0

        # LABELS
        def load_label_py(path):
            path = path.numpy().decode("utf-8")  # превращаем tf.Tensor -> bytes -> str
            boxes = []
            if os.path.exists(path):
                with open(path) as f:
                    for line in f.readlines():
                        values = list(map(float, line.strip().split()))
                        boxes.append(polygon_to_bbox(values))
            if len(boxes) < max_boxes:
                boxes += [[0,0,0,0,0]] * (max_boxes - len(boxes))
            else:
                boxes = boxes[:max_boxes]
            return np.array(boxes, dtype=np.float32)

        labels = tf.py_function(load_label_py, [label_path], tf.float32)
        labels.set_shape([max_boxes, 5])

        return img, labels


    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return ds


def write_text_to_image(img, text, pos, score=''):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Шрифт с поддержкой кириллицы
    font = ImageFont.truetype("DejaVuSans.ttf", 13)

    # Рисуем текст
    draw.text(pos, '{} {:.2f}'.format(text, score), font=font, fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# Функции детекции объектов и отображения предсказанной рамки
def draw_outputs(img, outputs, class_names, white_list=None):
    boxes, score, classes, nums = outputs # распознанные объекты
    boxes, score, classes, nums = boxes[0], score[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2]) # предсказанные ширина и высота
    for i in range(nums):
        # Отображаем объекты только из white_list
        if class_names[int(classes[i])] not in white_list:
            continue

        # Предсказанные координаты нижнего левого и правого верхнего углов
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))

        # Рисуем прямоугольник по двум предсказанным координатам
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 1)

        # Выводим имя класса предсказанного объекта и оценку
        img = cv2.putText(img, '{} {:.2f}'.format(
            class_names[int(classes[i])], score[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
    return img


def get_random_image_label():
    test_dir = "chess_yolo/test"
    img_dir = os.path.join(test_dir, 'images', '*.jpg')
    label_dir = os.path.join(test_dir, 'labels')
    files = glob.glob(img_dir)
    img_path = random.choice(files)

    label_name = f"{img_path.split('/')[-1].rsplit('.', 1)[0]}.txt"
    label_path = os.path.join(label_dir, label_name)

    return img_path, label_path


def detect_objects(yolo, img_path, return_img = True, white_list=None):
    image = img_path # путь к файлу
    img = tf.image.decode_image(open(image, 'rb').read(), channels=3) # загружаем изображение как тензор

    img = tf.expand_dims(img, 0) # добавляем размерность
    img = preprocess_image(img, settings.SIZE) # ресайзим изображение
    boxes, scores, classes, nums = yolo.predict(img) # делаем предсказание

    img = cv2.imread(image) # считываем изображение как картинку, чтобы на нем рисовать
    # Отрисовываем на картинке предсказанные объекты
    img = draw_outputs(img, (boxes, scores, classes, nums), settings.CLASS_NAMES, white_list)
    if return_img:
        return img
    # Сохраняем изображения с предсказанными объектами
    img_path = 'test.jpg'
    cv2.imwrite('detected_{:}'.format(img_path), img)

    # Открываем сохраненные изображения и выводим на экран
    detected = Image.open('detected_{:}'.format(img_path))
    detected.show()
    plt.title('Предсказанное изображение')
    plt.imshow(img)


def draw_yolo_labels(image_path, label_path, class_names=None):
    # Загружаем изображение
    image = image_path  # путь к файлу
    img = tf.image.decode_image(open(image, 'rb').read(), channels=3)  # загружаем изображение как тензор

    wh = np.flip(img.shape[0:2])
    # Читаем аннотации
    with open(label_path, "r") as f:
        lines = f.readlines()

    img = cv2.imread(image_path)
    for line in lines:
        parts = line.strip().split()

        cls_id = int(parts[0])
        polygon = list(map(float, parts[1:]))

        x1, y1, x2, y2, cls_id = polygon_to_bbox([cls_id] + polygon)
        x1y1 = tuple((np.array([x1, y1]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(x2, y2) * wh).astype(np.int32))

        # Рисуем прямоугольник
        color = (0, 255, 0)
        img = cv2.rectangle(img, x1y1, x2y2, color, 1)
        text = class_names[int(cls_id)]
        img = write_text_to_image(img, text, x1y1, 1.0)

    return img


def load_darknet_weights(model, weights_file):

    wf = open(weights_file, 'rb') # загружаем файл

    # Читаем из файла по элементам (первые 5, версия файла)
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    layers = settings.YOLO_V3_LAYERS # слои для загрузки

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)      # извлекаем блоки слоев из модели по имени
        for i, layer in enumerate(sub_model.layers): # пробегаемся по отдельным слоям блоков
            if not layer.name.startswith('conv2d'):  # пропускаем не сверточные слои
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                sub_model.layers[i + 1].name.startswith('batch_norm'):
                    batch_norm = sub_model.layers[i + 1]  # фиксируем если за слоем будет батч-нормализация



            filters = layer.filters                   # фильтров в слое
            size = layer.kernel_size[0]               # размер ядра в слое
            #in_dim = layer.input_shape[-1]           # input_shape в слоях Conv2d больше не поддерживается
            in_dim = layer.get_weights()[0].shape[2]  # входная размерность слоя

            # Вспоминаем структуру DBL: если нет нормализации, то добавляется смещение
            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters) # считываем веса для смещения
            else:
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4*filters) # считываем веса для нормализации
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]     # меняем форму

            conv_shape = (filters, in_dim, size, size)    # размерности сверточного слоя
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.prod(conv_shape))  # считываем веса для сверточного слоя
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0]) # меняем форму данных весов, решейпим и транспонируем

            # Если нет нормализации, то добавляем веса + смещение
            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                # Если есть нормализации, то добавляем веса в сверточный слой и в следующий за ним слой нормализации
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read weights'   # генерируем исключение, если файл не читается
    wf.close()
