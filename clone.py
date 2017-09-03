import csv
import cv2
import numpy as np
import sklearn
from random import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D

correction = 0.2
samples = []
with open('new_data/driving_log.csv') as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile)
    for line in reader: 
        # each line in csv file is augmented to six samples: center, left, right cameras, 
        # and mirrored versions of these three.
        # each sample is represented by a tuple: (file_name, measurement, is_mirrored)
        samples.append((line[0], float(line[3]), False))
        samples.append((line[1], float(line[3]) + correction, False))
        samples.append((line[2], float(line[3]) - correction, False))
        samples.append((line[0], float(line[3]), True))
        samples.append((line[1], float(line[3]) + correction, True))
        samples.append((line[2], float(line[3]) - correction, True))

# split the samples into training and validation data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'new_data/IMG/' + batch_sample[0].split('/')[-1]
                image = cv2.imread(name)
                angle = float(batch_sample[1])

                if batch_sample[2]: # if mirrored, flip the image and the sign of measurement
                    image = cv2.flip(image, 1)
                    angle = - angle
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# the model is based on nvidia model with modifications
# input image is cropped to ROI
# dropout layers are added to avoid overfitting
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping = ((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation = "relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation = "relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation = "relu"))
model.add(Convolution2D(64, 3, 3, activation = "relu"))
model.add(Convolution2D(64, 3, 3, activation = "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

print('Start training...')
model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(train_generator, samples_per_epoch = len(train_samples), \
    validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch = 10)
print('Finished training...')

model.save('model.h5')