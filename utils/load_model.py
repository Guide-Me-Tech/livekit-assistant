import tensorflow as tf

# from keras.models import Sequential, model_from_json
# from keras.preprocessing.sequence import pad_sequences
import os

maxlen = 500
batch_size = 64


# def preprocess_sentence(sss):
#    test_comments = [sss]
#    test_comments_category = ["command"]
#    embedding_name = "sentence_prediction/data/default"
#    pos_tags_flag = True
#    x_test, _, y_test, _ = encode_data(
#        test_comments,
#        test_comments_category,
#        data_split=1.0,
#        embedding_name=embedding_name,
#        add_pos_tags_flag=pos_tags_flag,
#    )
#    x_test = pad_sequences(x_test, maxlen=maxlen)
#    return x_test


import os

# import shutil
import pandas as pd

import tensorflow as tf

# import tensorflow_hub as hub
import tensorflow_text as text

# import seaborn as sns
# from pylab import rcParams

# import matplotlib.pyplot as plt
tf.get_logger().setLevel("ERROR")

# sns.set(style='whitegrid', palette='muted', font_scale=1.2)
# HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
# sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
# rcParams['figure.figsize'] = 12, 8
import warnings

warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelBinarizer
import keras


def load_model():
    model = keras.layers.TFSMLayer(
        "sentence_prediction/trained/tf/saved_model", call_endpoint="serving_default"
    )
    # model = tf.keras.models.load_model(
    #     "sentence_prediction/trained/" + "tf/saved_model"
    # )

    return model
