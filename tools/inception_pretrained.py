import sys, argparse
import tensorflow as tf
import loader as loader

from nets.googlenet import GoogLeNet

sys.path.append('../')

PRETRINED_PATH = '../parameters/googlenet.npy'
DATA_PATH = '../data/HKU-IS/'
IM_CHANNEL = 3


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_path', type=str, default=PRETRINED_PATH,
                        help='Directory of pretrain model')
    parser.add_argument('--im_name', type=str, default='.png',
                        help='Part of image name')
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                        help='Directory of test images')

    return parser.parse_args()


def test_pre_trained():
    FLAGS = get_args()
    label_dict = loader.load_label_dict()
    image_data = loader.read_image(
        im_name=FLAGS.im_name, n_channel=IM_CHANNEL,
        data_dir=FLAGS.data_path, batch_size=1)

    # Create model
    test_model = GoogLeNet(
        n_channel=IM_CHANNEL, n_class=1000, pre_trained_path=FLAGS.pretrained_path)
    test_model.create_test_model()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while image_data.epochs_completed < 1:
            batch_data = image_data.next_batch_dict()
            batch_file_name = image_data.get_batch_file_name()[0]
            pred = sess.run(test_model.layers['top_5'],
                            feed_dict={test_model.image: batch_data['image']})
            # display
            for re_prob, re_label, file_name in zip(pred[0], pred[1], batch_file_name):
                print('===============================')
                print('[image]: {}'.format(file_name))
                for i in range(5):
                    print('{}: probability: {:.02f}, label: {}'
                          .format(i + 1, re_prob[i], label_dict[re_label[i]]))


if __name__ == "__main__":
    test_pre_trained()
