import sys
import numpy as np
import skimage.transform

from dataflow.images import Image
from dataflow.cifar import CIFAR

sys.path.append('../')


def load_label_dict(dataset='imagenet'):
    label_dict = {}
    if dataset == 'cifar':
        file_path = '../data/cifarLabel.txt'
    else:
        file_path = '../data/imageNetLabel.txt'
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            if dataset == 'cifar':
                names = line.rstrip()
            else:
                names = line.rstrip()[10:]
            label_dict[idx] = names
    return label_dict


def read_image(im_name, n_channel, data_dir='', batch_size=1, rescale=True):
    def rescale_im(im):
        im = np.array(im)
        h, w = im.shape[0], im.shape[1]
        if h >= w:
            new_w = 224
            im = skimage.transform.resize(im, (int(h * new_w / w), 224),
                                          preserve_range=True)
        else:
            new_h = 224
            im = skimage.transform.resize(im, (224, int(w * new_h / h)),
                                          preserve_range=True)
        return im.astype('uint8')

    if rescale:
        pf_fnc = rescale_im
    else:
        pf_fnc = None

    image_data = Image(
        im_name=im_name,
        data_dir=data_dir,
        n_channel=n_channel,
        shuffle=False,
        batch_dict_name=['image'],
        pf_list=pf_fnc)
    image_data.setup(epoch_val=0, batch_size=batch_size)

    return image_data


def load_cifar(cifar_path, batch_size=64, subtract_mean=True):
    train_data = CIFAR(
        data_dir=cifar_path,
        shuffle=True,
        batch_dict_name=['image', 'label'],
        data_type='train',
        channel_mean=None,
        subtract_mean=subtract_mean,
        augment=True,
    )
    train_data.setup(epoch_val=0, batch_size=batch_size)

    valid_data = CIFAR(
        data_dir=cifar_path,
        shuffle=False,
        batch_dict_name=['image', 'label'],
        data_type='valid',
        channel_mean=train_data.channel_mean,
        subtract_mean=subtract_mean,
        augment=False,
    )
    valid_data.setup(epoch_val=0, batch_size=batch_size)

    return train_data, valid_data
