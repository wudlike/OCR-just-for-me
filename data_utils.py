import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 304875
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 33876
VOCUBLARY_SIZE = 5808


# 读取数据并保存在一个字典里
def read_text(filename):
    vocublary = {}
    vocublary['UNK'] = 0
    vocublary['SEQUENCE_START'] = 1
    vocublary['SEQUENCE_END'] = 2
    index = 3
    f = open(filename)
    line = f.readline()
    for s in line:
        vocublary[s] = index
        index = index + 1
    f.close()
    print('size of the vocublary: ', len(vocublary))
    return vocublary


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def one_hot(list):
    '''
    将label转换成one-hot形式
    ＠param list: label list
    ＠return: one-hot label
    '''
    vector = np.zeros(shape=[VOCUBLARY_SIZE])
    for index in list:
        vector[index] = 1
    return vector


# 生成TFRecord
def generation_TFRecord(data_base_dir, vocub_file_name):
    vocublary = read_text(vocub_file_name)

    image_name_list = []
    for file in os.listdir(data_base_dir):
        if file.endswith('.jpg'):
            image_name_list.append(file)

    random.shuffle(image_name_list)
    image_capacity = len(image_name_list)
    print(len(image_name_list))

    # 生成train tfrecord
    train_writer = tf.python_io.TFRecordWriter('./dataset/train_dataset.tfrecords')
    train_image_name_list = image_name_list[0:int(image_capacity * 0.9)]
    print(len(train_image_name_list))

    for train_image_name in train_image_name_list:
        train_image_label = []
        for s in train_image_name.strip('.jpg'):
            train_image_label.append(vocublary[s])
        train_image_label = one_hot(train_image_label)
        train_image_label = map(int, train_image_label)

        img = Image.open(os.path.join(data_base_dir, train_image_name))
        img = img.convert('RGB')
        # width, height = img.size
        img_array = np.asarray(img, np.uint8)
        shape = np.array(img_array.shape, np.int32)
        # 将图片转换为二进制形式
        img_raw = img.tobytes()
        # Example对象对label和image进行封装
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': int64_list_feature(train_image_label),
            'img_raw': bytes_feature(img_raw),
            'h': int64_feature(shape[0]),
            'w': int64_feature(shape[1]),
            'c': int64_feature(shape[2])
        }))
        # 序列转换成字符串
        train_writer.write(example.SerializeToString())
    train_writer.close()

    # 生成test tfrecord
    test_writer = tf.python_io.TFRecordWriter('./dataset/test_dataset.tfrecords')
    test_image_name_list = image_name_list[int(image_capacity * 0.9):image_capacity]
    print(len(test_image_name_list))
    for test_image_name in test_image_name_list:
        test_image_label = []
        for s in test_image_name.strip('.jpg'):
            test_image_label.append(vocublary[s])
        # 将label转变成one-hot形式，并将float类型转换成int类型
        test_image_label = one_hot(test_image_label)
        test_image_label = map(int, test_image_label)

        img = Image.open(os.path.join(data_base_dir, test_image_name))
        # 将图片转换成彩色图像，因为图像并不都是彩色图像
        img = img.convert('RGB')
        # width, height = img.size
        img_array = np.asarray(img, np.uint8)
        # print(img_array.shape)
        shape = np.array(img_array.shape, np.int32)
        # print(shape[0], shape[1], shape[2])
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': int64_list_feature(test_image_label),
            'img_raw': bytes_feature(img_raw),
            'h': int64_feature(shape[0]),
            'w': int64_feature(shape[1]),
            'c': int64_feature(shape[2])
        }))
        test_writer.write(example.SerializeToString())
    test_writer.close()


# 读取tfrecord文件
def read_and_decode(filename, max_width, fixed_height, batch_size, train=True):
    # 生成一个queue队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    # 将image数据和label取出来
    features = tf.parse_single_example(serialized=serialized_example,
                                       features={'label': tf.FixedLenFeature([VOCUBLARY_SIZE], tf.int64),
                                                 'img_raw': tf.FixedLenFeature([], tf.string),
                                                 'h': tf.FixedLenFeature([], tf.int64),
                                                 'w': tf.FixedLenFeature([], tf.int64),
                                                 'c': tf.FixedLenFeature([], tf.int64)
                                                 })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.cast(img, tf.float32)

    height = tf.cast(features['h'], tf.int32)
    width = tf.cast(features['w'], tf.int32)
    channel = tf.cast(features['c'], tf.int32)

    img = tf.reshape(img, shape=[height, width, channel])
    # 将图片reshape成固定大小
    resized_img = tf.image.resize_image_with_crop_or_pad(img,
                                                         target_height=fixed_height,
                                                         target_width=max_width)
    resized_img = tf.reshape(resized_img, shape=[fixed_height, max_width, 3])

    # 在流中抛出label张量
    label = tf.cast(features['label'], tf.int32)

    # shuffle batch
    min_fraction_of_examples_in_queue = 0.4
    if train is True:
        min_queue_examples = int(min_fraction_of_examples_in_queue * NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
        train_img_batch, train_label_batch = tf.train.shuffle_batch([resized_img, label],
                                                                    batch_size=batch_size,
                                                                    capacity=min_queue_examples + 3 * batch_size,
                                                                    min_after_dequeue=min_queue_examples,
                                                                    num_threads=32)
        return train_img_batch, train_label_batch
    else:
        min_queue_examples = int(min_fraction_of_examples_in_queue * NUM_EXAMPLES_PER_EPOCH_FOR_TEST)
        test_img_batch, test_label_batch = tf.train.batch([resized_img, label],
                                                          batch_size=batch_size,
                                                          capacity=min_queue_examples + 3 * batch_size,
                                                          num_threads=32)
        return test_img_batch, test_label_batch


# def CNN_VGG(inputs):
#     ''' CNN extract feature from each input image, 网络架构选择的是VGG(CRNN)
#     @param inputs: the input image
#     @return: feature maps
#     '''
#     with tf.variable_scope('VGG_CNN'):
#         #
#         conv1_1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=(3, 3),
#                                    padding='SAME', activation=tf.nn.relu, name='conv_1_1')
#         conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=(3, 3),
#                                    padding='SAME', activation=tf.nn.relu, name='conv_1_2')
#         pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=(2, 2), strides=2, name='pool_1')
#         #
#         conv2_1 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3),
#                                    padding='SAME', activation=tf.nn.relu, name='conv_2_1')
#         conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=(3, 3),
#                                    padding='SAME', activation=tf.nn.relu, name='conv_2_2')
#         pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=(2, 2), strides=2, name='pool_2')
#         #
#         conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=(3, 3),
#                                  padding='SAME', activation=tf.nn.relu, name='conv_3')
#         conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=(3, 3),
#                                  padding='SAME', activation=tf.nn.relu, name='conv_4')
#         pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=(1, 2), strides=2, name='pool_3')
#         #
#         conv5_1 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=(3, 3),
#                                    padding='SAME', activation=tf.nn.relu, name='conv5_1')
#         conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=(3, 3),
#                                  padding='SAME', activation=tf.nn.relu, name='conv5_2')
#         bn1 = tf.layers.batch_normalization(conv5_2, name='bn1')
#         pool4 = tf.layers.max_pooling2d(inputs=bn1, pool_size=(1, 2), strides=2, name='pool_4')
#         #
#         conv6 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=(3, 3),
#                                  padding='SAME', activation=tf.nn.relu, name='conv_6')
#         bn2 = tf.layers.batch_normalization(conv6, name='bn_2')
#         pool5 = tf.layers.max_pooling2d(inputs=bn2, pool_size=(1, 2), strides=2, name='pool_5')
#
#         conv7 = tf.layers.conv2d(inputs=pool5, filters=512, kernel_size=(3, 3),
#                                  padding='SAME', activation=tf.nn.relu, name='conv_7')
#     return conv7


# def main(argv):
#     generation_TFRecord('./train_val_dataset', './out.txt')
#     test_img, test_label = read_and_decode('./dataset/test_dataset.tfrecords', 1000, 64, 32, train=False)
#
#     # logits = CNN_VGG(test_img)
#     # print(logits.shape)
#     # logits = tf.layers.flatten(logits)
#     # print(logits.shape)
#     # weight = tf.Variable(tf.random_uniform([15872, 5805], 0.0, 1.0), dtype=tf.float32)
#     # logits = tf.matmul(logits, weight)
#     # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label)
#     # cross_entropy_loss = tf.reduce_mean(cross_entropy)
#     # optimizer = tf.train.AdadeltaOptimizer(0.001).minimize(cross_entropy_loss)
#
#     with tf.Session() as session:
#         session.run(tf.group(tf.global_variables_initializer(),
#                              tf.local_variables_initializer()))
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess=session, coord=coord)
#         batch_image, batch_label = session.run([test_img, test_label])
#         print(batch_image.shape, batch_label.shape)
#         # for index in range(10):
#         #     _, loss = session.run([optimizer, cross_entropy_loss])
#         #     print('loss:', loss)
#
#         coord.request_stop()
#         coord.join(threads=threads)
#
#
# if __name__ == '__main__':
#     tf.app.run()
