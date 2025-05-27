
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"      # Dual GPU
import tensorflow as tf
from keras import layers, models
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set parameters
path = 'E:/model/tumor_model/'

dicClass = {'nottumor': 0,
            'tumor': 1}

classnum = 2
EPOCHS = 30

# The following do not need to be changed
labelList = []
INIT_LR = 0.0001
Batch_Size = 128
np.random.seed(123)
datapath = path + 'train'

# Load images
# Unlike previous methods, here we do not process the images, but only return a list of image paths
def loadImageData():
    imageList = []
    listClasses = os.listdir(datapath)    # Class folders
    print(listClasses)
    for class_name in listClasses:
        label_id = dicClass[class_name]
        class_path = os.path.join(datapath, class_name)
        image_names = os.listdir(class_path)
        for image_name in image_names:
            image_full_path = os.path.join(class_path, image_name)
            labelList.append(label_id)
            imageList.append(image_full_path)
    return imageList

print("Starting to load data")
imageArr = loadImageData()
labelList = np.array(labelList)
print("Data loading complete")

# Random split
trainX, valX, trainY, valY = train_test_split(imageArr, labelList, test_size=0.2, random_state=123)

# Define image processing method
def generator(file_pathList, labels, batch_size, train_action=False):
    L = len(file_pathList)
    while True:
        input_labels = []
        input_samples = []
        for row in range(0, batch_size):
            temp = np.random.randint(0, L)
            X = file_pathList[temp]
            Y = labels[temp]
            image = cv2.imdecode(np.fromfile(X, dtype=np.uint8), -1)
            if image.shape[2] > 3:
                image = image[:, :, :3]
            # if train_action:
            #     image = train_transform(image=image)['image']
            # else:
            #     image = val_transform(image=image)['image']
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            image = img_to_array(image)
            input_samples.append(image)
            input_labels.append(Y)
        batch_x = np.asarray(input_samples)
        batch_y = np.asarray(input_labels)
        yield (batch_x, batch_y)

# Construct the network model
model = models.Sequential([
    layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(512, 512, 3)),
    layers.MaxPooling2D(2,2),
    # layers.Dropout(0.5),
    
    layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(2,2),
    # layers.Dropout(0.5),
    
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(2,2),
    # layers.Dropout(0.5),
    
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(2,2),
    # layers.Dropout(0.5),
    
    # layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    # layers.MaxPooling2D(2,2),
    # layers.Dropout(0.5),
     
    layers.Flatten(),
    # layers.Dense(256, activation='relu'),
    # layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(classnum, activation='softmax'),
])
 
# View the network structure
model.summary()

# Define loss function and optimizer
model.compile(optimizer=Adam(learning_rate=INIT_LR),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=path + 'best_model.hdf5',
                               monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

reduce = ReduceLROnPlateau(monitor='val_accuracy', patience=10,
                           verbose=1,
                           factor=0.5,
                           min_lr=1e-6)

# Model training
history = model.fit(generator(trainX, trainY, Batch_Size, train_action=True),
                    steps_per_epoch=len(trainX) / Batch_Size,
                    validation_data=generator(valX, valY, Batch_Size, train_action=False),
                    epochs=EPOCHS,
                    validation_steps=len(valX) / Batch_Size,
                    callbacks=[checkpointer, reduce])

model.save(path + 'my_model.h5')
print(history)

# Save training results and generate plots
loss_trend_graph_path = path + 'WW_loss.png'
acc_trend_graph_path = path + 'WW_acc.png'
print("Starting to plot")
# Summarize history for accuracy
fig = plt.figure(1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig(acc_trend_graph_path)
plt.close(1)
# Summarize history for loss
fig = plt.figure(2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig(loss_trend_graph_path)
plt.close(2)
print("Plotting complete")