import os  ; os.system("cls")
import keras, cv2, numpy
from keras.datasets.mnist import load_data
(features_train, labels_train), (features_test, labels_test) = load_data()
labels_train = numpy.identity(10)[labels_train]  # 10x10 identity matrix

def show_computation_methods():
	import tensorflow
	physical_devices = tensorflow.config.list_physical_devices()
	print("Physical devices:")
	for device in physical_devices:
	    print(device)


def show_image(data):
	image_rgb = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
	cv2.imshow("image", image_rgb)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



model = keras.Sequential([
	keras.layers.InputLayer((28, 28)),  # img size
	keras.layers.Flatten(),				# 784 element array
	keras.layers.Dense(100, "relu"),    # 100 neurons, w/ ea. 1 bias  [78 400 + 100]
	keras.layers.Dense(10, "softmax"),  # 10 output characters 0-9
])

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

model.fit(features_train, labels_train)
