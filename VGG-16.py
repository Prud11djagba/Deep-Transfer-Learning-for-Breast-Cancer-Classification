"""An implementation of a VGG16 for breast cancer classification

The original code has been taken from:
https://github.com/Staritino/Deep-Learning-for-Breast-Cancer-Prediction/blob/master/DDSMVgg16-script.py"""





from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import optimizers
from keras import backend as k
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
import numpy



#Check Directory
#Adjust samples, epochs, batch size


img_width, img_height = 50, 50
train_data_dir = '/home/aimsgh-02/Music/split_data/with_val/train'
validation_data_dir = '/home/aimsgh-02/Music/split_data/with_val/valid'
test_data_dir = '/home/aimsgh-02/Music/split_data/with_val/test'
train_samples = 194266
validation_samples = 41629
test_samples = 41629
epochs = 20
batch_size = 50

#Model
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(50, 50, 3))
model = Sequential()
model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model = Model(inputs=base_model.input, outputs=model(base_model.output))
for layer in base_model.layers:
    layer.trainable = False
print('base_model is now NOT trainable so we can ')


model.compile(optimizer=SGD(lr=0.001, momentum=0.9),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(3*" model compiled ouhaaaaaaaaa !!!\n")


train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

test_datagen = ImageDataGenerator(
rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size,
class_mode = "sparse", shuffle=True)

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "sparse", shuffle=True)

test_generator = test_datagen.flow_from_directory(
test_data_dir,
target_size = (img_height, img_width),
class_mode = "sparse", shuffle=False)


import time

# Start the timer
start_time = time.time()


model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples//batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_samples//batch_size)


for layer in model.layers[:15]:
    layer.trainable = False
for layer in model.layers[15:]:
    layer.trainable = True
model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=["accuracy"])


history= model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples//batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_samples//batch_size)

end_time = time.time()  
    
# Calculate the running time
running_time = end_time - start_time

# Print or log the running time
print(f"Running time: {running_time:.2f} seconds")



##Plot Accuracy
import matplotlib.pyplot as plt
print(history.history.keys())

plt.figure()
plt.plot(history.history['accuracy'],'gray',label='Training accuracy')
plt.plot(history.history['val_accuracy'],'purple',label='Validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
saveto = '/home/aimsgh-02/AIMS/Courses/AIMS_Gh-Research-Project_Template/images/vgg'
#plt.show
plt.savefig(saveto + '_accuracy_evol.png')



plt.figure()
plt.plot(history.history['loss'],'magenta',label='Training loss')
plt.plot(history.history['val_loss'],'brown',label='Validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()

#plt.show
plt.savefig(saveto + '_loss_evol.png')



y_true = test_generator.classes
predict = model.predict_generator(test_generator,test_generator.samples / test_generator.batch_size )
#y_pred = predict > 0.5
#Y_pred = model.predict_generator(test_generator)
y_pred = np.argmax(predict, axis=1)
print('Confusion Matrix')
print(confusion_matrix(y_true, y_pred))
print('Classification Report')
class_labels = list(test_generator.class_indices.keys())
print(class_labels)
#target_names = ['Normal', 'Abnormal']
print(classification_report(y_true, y_pred, target_names=class_labels))
accuracy = metrics.accuracy_score(y_true, y_pred)
print('Accuracy: ',accuracy)


print(len(y_true))



test_loss, test_acc = model.evaluate_generator(test_generator, test_generator.samples / test_generator.batch_size, verbose=1)
print('test acc:', test_acc)



