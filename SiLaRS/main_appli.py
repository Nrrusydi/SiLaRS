import datetime
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from scipy import stats
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow.keras import layers, models



mp_holistic = mp.solutions.holistic #Holistic model
mp_drawing = mp.solutions.drawing_utils #Drawing utilities
actions = np.array(['hello', 'terima kasih', 'saya', 'suka', 'awak', 'maaf', 'sama-sama', 'selamat berkenalan', 'tolong', 'jumpa lagi', 'R','U','S','Y','D','I']) #actions that we try to detect

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #color conversion bgr2rgb
    image.flags.writeable = False   #image is no longer writeable
    results = model.process(image)  #make prediction
    image.flags.writeable = True    #Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #color conversion rgb2gbr
    return image, results

def draw_styled_landmarks(image, results):

    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) #draw face connections
    
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )  #draw pose connections

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) #draw left hand connections
      
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) #draw right hand connections

def extract_keypoints(results): #extract the keypoint in landmark 
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def exitWindow():
   cap.release()
   cv2.destroyAllWindows()
   root.destroy()
   root.quit()  

model = models.Sequential()
model.add(layers.Conv1D(64, 3, activation='relu', input_shape=(30, 1662)))
model.add(layers.Conv1D(128, 3, activation='relu'))
model.add(layers.Conv1D(64, 3, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(actions.shape[0], activation='softmax'))

model.load_weights('sign_lang_new.h5')

cap= cv2.VideoCapture(0)
if (cap.isOpened() == False):
  print("Unable to read camera feed")

sequence = []
sentence = []
predictions = []
threshold = 0.5

def exitWindow():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()
    root.quit()  

while True:
    root = tk.Tk()
    root.title('Sign Language Recognition System')
    root.minsize(646, 530)
    root.maxsize(646, 530)
    root.configure(bg='#58F')

    # Initialize OpenCV video capture
    cap = cv2.VideoCapture(0)  # Replace 0 with the path to your video source if needed

    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Create GUI widgets
    frame_label = Label(root, bg='red')
    frame_label.pack()

    sentence_label = Label(root, text='', bg='#58F', fg='white', font=('Helvetica', 18))
    sentence_label.pack(side=tk.TOP)

    exit_button = Button(root, fg='white', bg='red', activebackground='white', activeforeground='red', text='EXIT ❌ ', relief=tk.RIDGE, height=2, width=20, command=exitWindow)
    exit_button.pack(side=tk.BOTTOM)

    # Function to update the video feed
    def update_video_feed():
        ret, frame = cap.read()
        global sequence

        if ret:
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic 
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

                global sentence
                # 3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

            # Create a label to display the sentence

            updated_sentence = ' '.join(sentence)
            sentence_label.config(text=updated_sentence)
            
            # Convert OpenCV image to PIL format
            image_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_pil)
            photo = ImageTk.PhotoImage(image=image_pil)
            
            # Update the label widget with the new image
            frame_label.config(image=photo)
            frame_label.image = photo
        
        # Schedule the function to be called again after a delay
        root.after(10,update_video_feed)

        

    # Start the video feed update process
    update_video_feed()

    root.mainloop()
