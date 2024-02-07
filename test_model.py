from attention import AttentionLayer
import keras
import tensorflow as tf
from keras.utils import CustomObjectScope
from utils import *
from keras.models import *
from attention_with_context import AttentionWithContext

with CustomObjectScope({'AttentionLayer': AttentionLayer,'AttentionWithContext':AttentionWithContext}):
	json_file = open('model/conv_lstm.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	cnn_lstm_model = model_from_json(loaded_model_json)
	cnn_lstm_model.load_weights("model/model_0040-0.0033.h5", 'r')