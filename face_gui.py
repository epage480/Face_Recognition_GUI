import numpy as np
import tensorflow as tf
import cv2
import os
import tkinter as tk
from PIL import Image
from PIL import ImageTk
import time
import faceModel

#model_path = "Model_Saves/face_triplet_model.ckpt"
model_path = "model_save128/face_triplet_model2.ckpt"
database_path = "Facial_Database/"

# For face detection
face_cascade = cv2.CascadeClassifier('C:/Users/Eric/AppData/Local/conda/conda/pkgs/opencv3-3.1.0-py35_0/Library/etc/haarcascades/haarcascade_frontalface_default.xml')

# Set up camera for capture
cap = cv2.VideoCapture(0)

face = []

# Initialize placeholders
image = tf.placeholder(dtype=tf.float32, shape=[None, 250, 250, 3])
label = tf.placeholder(dtype=tf.int32, shape=[None])

# Initialize model
myModel = faceModel.faceModel(image, label)

# Start tensorflow session and restore model
sess = tf.Session()
sess.run(tf.global_variables_initializer())
myModel.restore(sess, model_path)

# Generate embedding of every image in database
labels     = [f for f in os.listdir(database_path) if f.lower().endswith(".jpg")]
files = [f for f in os.listdir(database_path) if f.lower().endswith(".jpg")]
embeddings = [sess.run(myModel.inference, feed_dict={image: cv2.imread(database_path+f).reshape((1,250,250,3))}) for f in files]

def check_face(face):
    img_embed = sess.run(myModel.inference, feed_dict={image: face.reshape((1, 250, 250, 3))})
    distances = []
    for embed in embeddings:
        distances.append(np.linalg.norm(img_embed - embed))

    if min(distances) < .6:
        msg.config(text="Access Granted", fg='green')

        # Read database file and convert to photoImage format
        frame = cv2.imread(labels[distances.index(min(distances))])

        # Convert to gui friendly format and display image
        facetk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)))
        frecLabel.imgtk = facetk
        frecLabel.configure(image=facetk)

    else:
        msg.config(text="Access Denied", fg='red')

    # Debugging Purposes only, prints top 5 labels and their distances
    #print()
    #for count in range(5):
    #    print(count, labels[distances.index(min(distances))], min(distances))
    #    del distances[distances.index(min(distances))]


def update_live_frame():
    start_time = time.time()
    # Read image from webcam and convert format
    _, frame = cap.read() # 480x640
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw bounding box and get faces
    global face
    face = []
    for count, (x, y, w, h) in enumerate(faces):
        face.append(frame[y:y + h, x:x + w])
        face[count] = cv2.resize(face[count], (250, 250))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if len(faces) == 1:
        #msg.config(text="One face detected", fg='green')
        # Convert the detected face to a gui-friendly format and show
        facetk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(face[0], cv2.COLOR_BGR2RGBA)))
        fdetectLabel.imgtk = facetk
        fdetectLabel.configure(image=facetk)


        check_face(face[0])

    # Only need to handle 1 face, error messages
    elif len(faces) == 0:
        msg.config(text="No face detected", fg='red')
    else:
        msg.config(text="Multiple faces detected", fg='red')

    # Get time elapsed and edit image to show frame-rate
    fr = 1/(time.time() - start_time)
    fr = str("%0.2f"%fr)
    cv2.putText(frame, fr, (545, 465), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Convert live frame and face to gui friendly format
    frametk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)))

    # Show webcam image in gui interface
    videoLabel.imgtk = frametk
    videoLabel.configure(image=frametk)

    videoLabel.after(10, update_live_frame)

def enroll_user():
    name = nameEntry.get()
    if os.path.isfile(database_path+name+".jpg"):
        msg.config(text="Name already enrolled")
    else:
        print("1")
        cv2.imwrite(database_path + name+".jpg", face[0])
        print("2")
        labels.append(database_path + name+".jpg")
        print("3")
        embeddings.append(sess.run(myModel.inference, feed_dict={image: face[0].reshape((1, 250, 250, 3))}))
        print("4")
        msg.config(text="Welcome to the club "+name)
        print("5")

# Set up main window for GUI
window = tk.Tk()
window.wm_title("Face Recognition")
#Slider window (slider controls stage position)
sliderFrame = tk.Frame(window, width=600, height=100)
sliderFrame.grid(row = 600, column=0, padx=10, pady=2)

# Frame for live video feed
videoFrame = tk.Frame(window)
videoFrame.grid(row=0, column=0, padx=2, pady=2)
videoLabel = tk.Label(videoFrame)
videoLabel.grid(row=0, column=0)

# Frame for detected face
fdetectFrame = tk.Frame(window)
fdetectFrame.grid(row=0, column=1, padx=2, pady=2)
fdetectLabel = tk.Label(fdetectFrame)
fdetectLabel.grid(row=0, column=1)

# Frame for recognized face
frecFrame = tk.Frame(window)
frecFrame.grid(row=1, column=1, padx=2, pady=2)
frecLabel = tk.Label(fdetectFrame)
frecLabel.grid(row=1, column=1)

# Frame for output text
msg = tk.Message(window, text = "Feedback Area Here")
msg.grid(row=3, column=0, padx=2, pady=2)
msg.config(bg='white', font=('times', 24))

# Frame for button
button = tk.Button(sliderFrame, text='Enroll New User', width=25, command=enroll_user)
button.grid(row=1, column=0, padx=2, pady=2)

# Frame for Name Entry
nameLabel = tk.Label(window, text="Input Terminal")
nameLabel.grid(row=2, column=0, padx=2, pady=2)
nameEntry = tk.Entry(nameLabel, bd =5)
nameEntry.grid(row=2, column=1, padx=2, pady=2)

update_live_frame()  #Display 2
window.mainloop()  #Starts GUI