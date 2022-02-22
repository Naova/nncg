#!/usr/bin/env python3
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
import tensorflow as tf
import argparse
from keras.models import load_model
from nncg.nncg import NNCG
from applications.daimler.loader import load_imdb

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
parser = argparse.ArgumentParser(description='Train the network given ')

parser.add_argument('-m', '--model-path', dest='model_path',
                    help='Store the trained model using this path. Default is model.h5.')
parser.add_argument('-c', '--code-path', dest='code_path',
                    help='Path where the file is to be stored. Default is current directory')

args = parser.parse_args()

model_path = "yolo_modele_simulation.h5"
code_path = "."


if args.model_path is not None:
    model_path = args.model_path

if args.code_path is not None:
    code_path = args.code_path

images = {}
images["mean"] = [1]
images["y"] = [0]
images["images"] = [np.zeros((120, 160, 3))]

model = load_model(model_path, compile=False)

sse_generator = NNCG()
sse_generator.keras_compile(images["images"], model, code_path, "vision", arch="sse3", testing=-1)
