import os
import sys
import argparse
import tensorflow as tf
import loader as loader

from nets.googlenet import GoogLeNet_cifar
from helper.trainer import Trainer
from helper.evaluator import Evaluator

sys.path.append('../')

DATA_PATH = '/home/maliqi/leo/vggnet/cifar-10-batches-py/'
SAVE_PATH = '../ckpt/save_path/'
PRETRINED_PATH = '/home/maliqi/leo/googlenet/googlenet.npy'

IM_PATH = '../data/cifar/'

device_num = 2
os.environ['CUDA_VISIBLE_DEVICES'] = str(device_num)
print('CUDA_VISIBLE_DEVICES:', device_num)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate the model')
    parser.add_argument('--predict', action='store_true',
                        help='Get prediction result')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine tuning the model')
    parser.add_argument('--load', type=int, default=99,
                        help='Epoch id of pre-trained model')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--bsize', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--keep_prob', type=float, default=0.4,
                        help='Keep probability for dropout')
    # parser.add_argument('--maxepoch', type=int, default=100,
    #                     help='Max number of epochs for training')
    parser.add_argument('--maxepoch', type=int, default=120,
                        help='Max number of epochs for training')

    parser.add_argument('--im_name', type=str, default='.png',
                        help='Part of image name')

    return parser.parse_args()


def train():
    FLAGS = get_args()
    train_data, valid_data = loader.load_cifar(
        cifar_path=DATA_PATH, batch_size=FLAGS.bsize, subtract_mean=True)

    pre_trained_path = None
    if FLAGS.finetune:
        pre_trained_path = PRETRINED_PATH

    train_model = GoogLeNet_cifar(
        n_channel=3, n_class=10, pre_trained_path=pre_trained_path,
        bn=True, wd=0, sub_imagenet_mean=False,
        conv_trainable=True, fc_trainable=True)
    train_model.create_train_model()

    valid_model = GoogLeNet_cifar(
        n_channel=3, n_class=10, bn=True, sub_imagenet_mean=False)
    valid_model.create_test_model()

    trainer = Trainer(train_model, valid_model, train_data, init_lr=FLAGS.lr)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(SAVE_PATH)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        for epoch_id in range(FLAGS.maxepoch):
            trainer.train_epoch(sess, keep_prob=FLAGS.keep_prob, summary_writer=writer)
            trainer.valid_epoch(sess, dataflow=valid_data, summary_writer=writer)
            saver.save(sess, '{}inception-cifar-epoch-{}'.format(SAVE_PATH, epoch_id))
        saver.save(sess, '{}inception-cifar-epoch-{}'.format(SAVE_PATH, epoch_id))
        writer.close()


def evaluate():
    FLAGS = get_args()
    train_data, valid_data = loader.load_cifar(
        cifar_path=DATA_PATH, batch_size=FLAGS.bsize, subtract_mean=True)
    valid_model = GoogLeNet_cifar(
        n_channel=3, n_class=10, bn=True, sub_imagenet_mean=False)
    valid_model.create_test_model()

    evaluator = Evaluator(valid_model)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '{}inception-cifar-epoch-{}'.format(SAVE_PATH, FLAGS.load))
        print('training set:', end='')
        evaluator.accuracy(sess, train_data)
        print('testing set:', end='')
        evaluator.accuracy(sess, valid_data)


def predict():
    FLAGS = get_args()
    label_dict = loader.load_label_dict(dataset='cifar')
    image_data = loader.read_image(
        im_name=FLAGS.im_name, n_channel=3,
        data_dir=IM_PATH, batch_size=1, rescale=False)

    test_model = GoogLeNet_cifar(
        n_channel=3, n_class=10, bn=True, sub_imagenet_mean=False)
    test_model.create_test_model()

    with tf.Session(config=tf_config) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '{}inception-cifar-epoch-{}'.format(SAVE_PATH, FLAGS.load))
        while image_data.epochs_completed < 1:
            batch_data = image_data.next_batch_dict()
            batch_file_name = image_data.get_batch_file_name()[0]
            pred = sess.run(test_model.layers['top_5'],
                            feed_dict={test_model.image: batch_data['image']})
            for re_prob, re_label, file_name in zip(pred[0], pred[1], batch_file_name):
                print('===============================')
                print('[image]: {}'.format(file_name))
                for i in range(5):
                    print('{}: probability: {:.02f}, label: {}'
                          .format(i + 1, re_prob[i], label_dict[re_label[i]]))


if __name__ == "__main__":
    FLAGS = get_args()

    if FLAGS.train:
        train()
    if FLAGS.eval:
        evaluate()
    if FLAGS.predict:
        predict()
