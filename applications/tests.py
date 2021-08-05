from nncg.nncg import NNCG
from applications.daimler.loader import random_imdb, load_images, finish_db
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Flatten, MaxPooling2D, Convolution2D, Dropout, Dense
from tensorflow.keras.models import Sequential
import argparse


def print_success(name):
    """
    Prints that a test has passed.
    :param name: Name of the test.
    :return: None.
    """
    print('''
    
######################################################################
            {} passed
######################################################################    
    
'''.format(name))


def no_dense():
    """
    Tests an example CNN with no dense layer.
    :return: None
    """
    num_imgs = 10
    nncg = NNCG()
    no_dense = Sequential()
    no_dense.add(Convolution2D(4, (3, 3), input_shape=(36, 18, 1),
                               activation='relu', padding='same'))
    no_dense.add(MaxPooling2D(pool_size=(2, 2)))
    no_dense.add(Convolution2D(8, (3, 3), padding='same', activation='relu', bias_initializer='random_uniform'))
    no_dense.add(MaxPooling2D(pool_size=(2, 2)))
    no_dense.add(Convolution2D(16, (3, 3), padding='same', activation='relu',
                               bias_initializer='random_uniform'))  # Could be softmax
    no_dense.add(MaxPooling2D(pool_size=(4, 2)))
    no_dense.add(Dropout(0.4))
    no_dense.add(Convolution2D(2, (2, 2), activation='softmax'))
    no_dense.add(Flatten())
    images = random_imdb(num_imgs, no_dense.input.shape[1:].as_list())
    nncg.keras_compile(images, no_dense, 'no_dense.c')
    print_success('no_dense')


def dense_model():
    """
    Tests an example CNN with a Dense layer and valid padding.
    :return: None.
    """
    num_imgs = 10
    nncg = NNCG()
    dense_model = Sequential()
    dense_model.add(Convolution2D(8, (3, 3), input_shape=(70, 50, 1),
                                  activation='relu', padding='same'))
    dense_model.add(MaxPooling2D(pool_size=(2, 2)))
    dense_model.add(Convolution2D(16, (3, 3), padding='valid', activation='relu', bias_initializer='random_uniform'))
    dense_model.add(MaxPooling2D(pool_size=(2, 2)))
    dense_model.add(Convolution2D(32, (3, 3), padding='valid', activation='relu', bias_initializer='random_uniform'))
    dense_model.add(MaxPooling2D(pool_size=(2, 2)))
    dense_model.add(Dropout(0.4))
    dense_model.add(Flatten())
    dense_model.add(Dense(2, activation='softmax'))
    images = random_imdb(num_imgs, dense_model.input.shape[1:].as_list())
    nncg.keras_compile(images, dense_model, 'dense_model.c')
    print_success('dense_model')


def strides():
    """
    Tests an example CNN with additional unusual strides.
    :return: None.
    """
    num_imgs = 10
    nncg = NNCG()
    strides = Sequential()
    strides.add(Convolution2D(4, (3, 3), input_shape=(101, 101, 1),
                              activation='relu', padding='same', strides=(3, 3)))
    strides.add(MaxPooling2D(pool_size=(2, 2)))
    strides.add(Convolution2D(8, (3, 3), padding='valid', activation='relu', strides=(2, 3)))
    strides.add(Convolution2D(16, (3, 3), padding='valid', activation='relu'))
    strides.add(Flatten())
    strides.add(Dense(2, activation='softmax'))
    images = random_imdb(num_imgs, strides.input.shape[1:].as_list())
    nncg.keras_compile(images, strides, 'strides.c')
    print_success('strides')


def VGG16_test():
    """
    Tests a full VGG16.
    :return: None.
    """
    num_imgs = 1
    nncg = NNCG()
    vgg16_m = VGG16(weights=None)
    db = random_imdb(num_imgs, vgg16_m.input.shape[1:].as_list())
    nncg.keras_compile(db, vgg16_m, 'vgg16.c', weights_method='stdio')
    print_success('VGG16')


def VGG16_quantized_test(db):
    """
    Tests a full quantized VGG16.
    :param db: Image database to test with real images.
    :return: None.
    """
    num_imgs = 1
    nncg = NNCG()
    vgg16_m = VGG16(weights='imagenet')
    nncg.keras_compile(db, vgg16_m, 'vgg16.c', weights_method='stdio', quatization=True, arch='sse3',
                       test_mode='classification')
    print_success('VGG16')


def VGG19_test():
    """
    Tests a full VGG19.
    :return: None.
    """
    num_imgs = 1
    nncg = NNCG()
    vgg19_m = VGG19(weights=None)
    db = random_imdb(num_imgs, vgg19_m.input.shape[1:].as_list())
    nncg.keras_compile(db, vgg19_m, 'vgg19.c', weights_method='stdio')
    print_success('VGG19')


def VGG19_quantized_test(db):
    """
    Tests a full VGG19 using quantization.
    :param db: Image database to test with real images.
    :return: None.
    """
    nncg = NNCG()
    vgg19_m = VGG19(weights='imagenet')
    nncg.keras_compile(db, vgg19_m, 'vgg19.c', weights_method='stdio', quatization=True, arch='sse3',
                       test_mode='classification')
    print_success('VGG19')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test using various CNN.')
    parser.add_argument('-i', '--image-folder', dest='img_path',
                        help='Path to the folder containing 0, 1 etc. folders with jpg images. '
                             'Default is not using real images for testing.')
    args = parser.parse_args()

    # All tests do not need an image database so we just call them.
    no_dense()
    dense_model()
    strides()
    VGG16_test()
    VGG19_test()
    # Testing quantied networks only makes sense with real images
    if args.img_path is not None:
        db = []
        for i in range(4):
            db = load_images(args.img_path + "/" + str(i) + "/*.JPEG", {"x": 224, "y": 224}, i, 4, db,
                             rotate=False, flip=False, gain=False, color=True)
        db, y, mean = finish_db(db, color=True)

        VGG16_quantized_test(db)
        VGG19_quantized_test(db)
