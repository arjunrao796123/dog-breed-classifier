from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras.preprocessing import image                  
from tqdm import tqdm
import tensorflow as tf

class Resnet50():

	def __init__(self):
		pass

	def res_set(self):
		self.ResNet50_model = ResNet50(weights='imagenet')
		#self.graph = tf.get_default_graph()

	def ResNet50_predict_labels(self,img_features):
	    # returns prediction vector for image located at img_path
	    img = preprocess_input(img_features)
	    return np.argmax(self.ResNet50_model.predict(img))

	### returns "True" if a dog is detected in the image stored at img_path
	def dog_detector(self,img_features):
	    prediction = self.ResNet50_predict_labels(img_features)
	    return ((prediction <= 268) & (prediction >= 151)) 

