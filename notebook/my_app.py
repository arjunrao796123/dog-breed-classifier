
import cv2                
import matplotlib.pyplot as plt 
from .resnet import Resnet50  
from .xception_model import DogXception               
import os      
import sys
from keras.preprocessing import image 
import getopt
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')



class DogId:
	def __init__(self):
		pass
	def compile(self):
		
		#face detector
		self.face_cascade = cv2.CascadeClassifier(os.path.join(master_path,'haarcascades','haarcascade_frontalface_alt.xml'))
		
		#dog detector
		self.resnet_dog = Resnet50()
		self.resnet_dog.res_set()

		#the trained xception model
		self.xception = DogXception()
		self.xception.compile(os.path.join(master_path,'saved_models','weights.best.Xc.hdf5'))

	def face_detected(self, image=None):
		self._check_features(image)
		img = cv2.imread(self.image_path)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = self.face_cascade.detectMultiScale(gray)
		return len(faces) > 0


	def load_image(self, file_path):
		img = image.load_img(file_path, target_size=(224, 224))
		self.image_path = file_path
		self.features = np.zeros((1,224,224,3))
		self.features[0,:,:,:] = image.img_to_array(img)

	def dog_detector(self, image=None):
		self._check_features(image)
		return self.resnet_dog.dog_detector(self.features)

	def dog_classify(self, image=None): 
		self._check_features(image)
		return self.xception.predict(self.features)


	def process(self, image=None):
		self._check_features(image)
		# check if dog 
		if self.dog_detector():
		    code = 1
		else:
		    # if not we can expect the image to be of a human 
		    # check if face is clearly identifiable
		    if self.face_detected():
		        code = 2
		    else:
		        code = 0
		        return (code, 'Neither a dog nor a human face' , {})

		# get predictions 
		predictions = self.dog_classify()
		if code == 2:
		    info = 'Human face'
		else:
		    info = 'Dog'

		return (code, info, predictions)

	def _check_features(self, image):
		if image != None:
			self.load_image(image)

		if type(self.features) == type(None):
			raise ValueError('Image not provided')

def main(argv):
		try:
		    opts, args = getopt.getopt(argv, '', ['file=', 'display='])
		    disp = 5
		    for o,a in opts:
		        if o == '--file':
		            file_path =  a
		        elif o == '--display':
		            disp = int(a)
		except getopt.GetoptError as err:
		    print(err)

		if file_path != None:
		    app = DogId()
		    app.compile()
		    code, info, predictions = app.process(file_path)
		    print(info)
		    if code == 0:
		        return

		    for i in range(disp):
		        print("It is a {}, with a probability of {:.3f}".format( predictions['breeds'][i], predictions['prob'][i] ))

		import gc; gc.collect()

master_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    main(sys.argv[1:])