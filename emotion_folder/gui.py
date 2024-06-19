import tkinter as tk
from tkinter import filedialog
from tkinter import *
from sklearn import metrics
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, ImageTk

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
       loaded_model_json = (file.read())
       model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
top=tk.Tk()
top.geometry('800x600')
top.title('Facial Expression Recognition')
top.configure(background='white')
label1=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model(1).json", "model_weights(1).h5")

EMOTIONS_LIST = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral', 'Sad', 'Surprise' ]

def Detect(file_path):
    global Label_packed
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.3, 5)
    try:
        for(x,y,w,h) in faces:
            fc=gray[y:y+h, x:x+w] # crop the face
            roi=cv2.resize(fc,(48,48)) # resize the image
            pred= EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))] # predict the emotion
            print("Predicted Emotion is : ", pred) # print the emotion
            label1.configure(foreground='#011638', text=pred) # display the emotion
    except:
        print("No face found")
        label1.configure(foreground='#011638', text="Unable to detect face") # if no face found

def show_Detect_button(file_path): # show detect button
    detect_b=Button(top, text="Detect",command=lambda: Detect(file_path),padx=10,pady=5) # detect button
    detect_b.configure(background='#364156', foreground='white',font=('arial',10,'bold')) # button design
    detect_b.place(relx=0.79,rely=0.46) # button placement

def upload_image():
    try:
        file_path=filedialog.askopenfilename() # open the file dialog
        uploaded=Image.open(file_path) # open the image
        uploaded.thumbnail(((top.winfo_width()/2.3),(top.winfo_height()/2.3))) # set the image size
        im=ImageTk.PhotoImage(uploaded) # photo image
        sign_image.configure(image=im) # set the image
        sign_image.image=im 
        label1.configure(text='') # clear the label
        show_Detect_button(file_path) # call the detect button
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5) # upload button 
### what is padx and pady in tkinter
# padx and pady are the attributes of the Button class in tkinter module. They are used to set the padding of the button.
# padx is used to set the padding in the x-axis and pady is used to set the padding in the y-axis.
# The padding is the space between the text and the border of the button.
# The default value of padx and pady is 1.
# The value of padx and pady can be set in pixels.
# The value of padx and pady can be set using the padx and pady attributes of the Button class.""""
upload.configure(background='#364156', foreground='white',font=('arial',20,'bold')) # button design
upload.pack(side=BOTTOM,pady=50) # button placement
sign_image.pack(side=BOTTOM,expand=True) # image placement
label1.pack(side=BOTTOM,expand=True) # label placement
heading = Label(top, text="Facial Expression Recognition",pady=20, font=('arial',25,'bold')) # heading
heading.configure(background='white',foreground='#364156') # heading design
heading.pack()
top.mainloop()



