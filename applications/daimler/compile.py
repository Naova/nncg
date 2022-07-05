#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
import tensorflow as tf
import argparse
from keras.models import load_model
from nncg.nncg import NNCG
from applications.daimler.loader import load_imdb

def compile(model_path, code_path, imgdb_path):
    images = load_imdb(imgdb_path)
    model = load_model(model_path, compile=False)

    general_generator = NNCG()
    ok = general_generator.keras_compile(images["images"], model, code_path, "qsse3", arch="sse3", testing=1000,
                                         quatization=True, test_mode='classification')
    assert(ok > 95)
    sse_generator = NNCG()
    ok = sse_generator.keras_compile(images["images"], model, code_path, "sse3", arch="sse3", testing=1000)
    assert(ok == 100)

if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    parser = argparse.ArgumentParser(description='Train the network given ')

    parser.add_argument('-b', '--database-path', dest='imgdb_path',
                        help='Path to the image database to use for training. '
                            'Default is img.db in current folder.')
    parser.add_argument('-m', '--model-path', dest='model_path',
                        help='Store the trained model using this path. Default is model.h5.')
    parser.add_argument('-c', '--code-path', dest='code_path',
                        help='Path where the file is to be stored. Default is current directory')

    args = parser.parse_args()

    imgdb_path = "img.db"
    model_path = "model.h5"
    code_path = "."

    if args.imgdb_path is not None:
        imgdb_path = args.imgdb_path

    if args.model_path is not None:
        model_path = args.model_path

    if args.code_path is not None:
        code_path = args.code_path

    compile(model_path, code_path, imgdb_path)

