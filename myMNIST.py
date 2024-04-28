import os  ; os.system("cls")
import keras, cv2, numpy, sys
from keras.datasets.mnist import load_data
(features_train, labels_train), (features_test, labels_test) = load_data()
labels_train = numpy.identity(10)[labels_train]  # 10x10 identity matrix

SAVE_WEIGHTS = False
LOAD_MODEL = True
DNN_TYPE = "CNN"  # DNN, RNN, CNN, ResNet
WEIGHTS_FILENAME = f"{sys.argv[0].replace('.py', '')}_{DNN_TYPE}.h5" 
CNN_LAYERS = 2 # if using CNN
NUM_EPOCHS = 2
TEST_IMAGE = cv2.cvtColor(cv2.imread("my-great-8.png"), cv2.COLOR_RGB2GRAY)

def show_computation_methods():
	import tensorflow
	physical_devices = tensorflow.config.list_physical_devices()
	print("Physical devices:")
	for device in physical_devices:
	    print(device)

def show_image(data):
    if len(data.shape) == 3 and data.shape[2] == 3:
        image_rgb = data
    else:
        image_rgb = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    cv2.imshow("image", image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if LOAD_MODEL:
	model = keras.models.load_model(WEIGHTS_FILENAME)
else:

	# SETUP

	model = keras.Sequential()
	model.add(keras.layers.InputLayer((28, 28)))  # img size

	if DNN_TYPE == "CNN":
		model.add(keras.layers.Reshape((28, 28, 1)))
		for layer in range(CNN_LAYERS):
			model.add(keras.layers.Conv2D(
				filters=32,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding="valid",
				activation="relu"
			))
			model.add(keras.layers.MaxPooling2D())

	model.add(keras.layers.Flatten())			  # 784 element array
	model.add(keras.layers.Dense(100, "relu"))    # 100 neurons, w/ ea. 1 bias  [78 400 + 100]
	model.add(keras.layers.Dense(10, "softmax"))  # 10 output characters 0-9


	model.compile(
		optimizer="adam", 
		loss="categorical_crossentropy",
		metrics=["accuracy"]
	)
	os.system("cls")
	show_computation_methods()
	model.summary()

	prediction = model.predict(features_train[[0]])
	for index, ele in enumerate(prediction[0]):
		print(f"{index}: {ele * 100:.0f} %")
		# SORT

	# TRAINING

	model.fit(features_train, labels_train, epochs=NUM_EPOCHS)
	if SAVE_WEIGHTS:
		model.save(WEIGHTS_FILENAME)

# TEST

test = model.predict(
	TEST_IMAGE.reshape(
		(
			1,  # batch dimension
			28, # height
			28, # width
		)  
	)
)  

for index, ele in enumerate(test[0]):
	print(f"{index}: {ele * 100:.0f} %")
	# SORT
