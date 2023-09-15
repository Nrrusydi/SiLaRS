#NAME : RUSYDI NASUTION

import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

DATA_PATH = os.path.join('FYP_Data2') #path for exported data, numpy arrays
actions = np.array(['hello', 'terima kasih', 'sama-sama', 'selamat berkenalan', 'tolong']) #actions that we try to detect
no_sequences = 30 #thirty videos worth of data
sequence_length = 30 #videos are going to be 30 frames in length
start_folder = 30 #folder start

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences) #put data into X

y = to_categorical(labels).astype(int) #put label into Y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05) #split dataset

log_dir = os.path.join('Logs2')
tb_callback = TensorBoard(log_dir=log_dir)

# Define the CNN model
model = models.Sequential()
model.add(layers.Conv1D(64, 3, activation='relu', input_shape=(30, 1662)))
model.add(layers.Conv1D(128, 3, activation='relu'))
model.add(layers.Conv1D(64, 3, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train, y_train, epochs=22, batch_size=32, validation_data=(X_test, y_test))

model.save('sign_lang_new2.h5')

pred = model.predict(X_test)

test_true = np.argmax(y_test, axis=1).tolist()
pred = np.argmax(pred, axis=1).tolist() #convert numpy array to python list

#confusion matric
mcm = multilabel_confusion_matrix(test_true, pred)
print("Multilabel Confusion Matrix:")
print(mcm)

#calculate accuray
accuracy = accuracy_score(test_true, pred)
print("Accuracy:", accuracy)

#calculate recall
recall = recall_score(test_true, pred, average='macro')  
print("Recall:", recall)

#calculate precision
precision = precision_score(test_true, pred, average='macro')  
print("Precision:", precision)

#calculate F1 score
f1 = f1_score(test_true, pred, average='macro')  
print("F1 Score:", f1)