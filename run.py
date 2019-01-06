import tensorflow as tf
import numpy as np
import argparse
from model import Model
import os

parse = argparse.ArgumentParser(description='Argument parser')

parse.add_argument('--phase', dest='phase', default='train', help='train, test')
parse.add_argument('--batch_size', dest='batch_size', type=int, default=1,
                   help='# images in batch')
parse.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parse.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint',
                   help='models are saved here')
parse.add_argument('--dropout_rate', dest='dropout_rate', default=0.5,
                   help='the rate of dropout')
parse.add_argument('--learning_rate', dest='learning_rate', default=0.0001, help='learning rate')
parse.add_argument('--train_dataset', dest='train_dataset', default='./dataset/train_dataset.tfrecords',
                   help='where the train dataset is')
parse.add_argument('--test_dataset', dest='test_dataset', default='./dataset/test_dataset.tfrecords',
                   help='where the test dataset is')
parse.add_argument('--image_max_width', dest='image_max_width', default=1000, type=int,
                   help='assume the max width of an image')
parse.add_argument('--image_fixed_height', dest='image_fixed_height', type=int, default=64,
                   help='the defined height of an image')

args = parse.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    model_instance = Model(batch_size=args.batch_size,
                           num_epoch=args.epoch,
                           dropout_rate=args.dropout_rate,
                           learning_rate=args.learning_rate,
                           train_dataset=args.train_dataset,
                           test_dataset=args.test_dataset,
                           image_max_width=args.image_max_width,
                           image_fixed_height=args.image_fixed_height
                           )
    if args.phase is 'train':
        model_instance.train()
    else:
        model_instance.test()


if __name__ == '__main__':
    tf.app.run()
