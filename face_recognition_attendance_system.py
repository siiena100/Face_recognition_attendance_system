import tkinter as tk
from tkinter import messagebox
import os
import cv2
import numpy as np
from PIL import Image
import mysql.connector
from datetime import datetime, timedelta

# Initialize the main window
window = tk.Tk()
window.title("Face Recognition Attendance System")

# Input fields for user details
l1 = tk.Label(window, text="Name", font=("Arial", 20))
l1.grid(column=0, row=0)
t1 = tk.Entry(window, width=50, bd=5)
t1.grid(column=1, row=0)

l2 = tk.Label(window, text="Roll", font=("Arial", 20))
l2.grid(column=0, row=1)
t2 = tk.Entry(window, width=50, bd=5)
t2.grid(column=1, row=1)

#l3 = tk.Label(window, text="Address", font=("Arial", 20))
#l3.grid(column=0, row=2)
#t3 = tk.Entry(window, width=50, bd=5)
#t3.grid(column=1, row=2)

# MySQL Connection
def get_database_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Swarna@0902",
        database="Authorized_user"
    )

# Function to generate dataset
def generate_dataset():
    if t1.get() == "" or t2.get() == "":
        messagebox.showinfo('Result', 'Please provide complete details of the user')
        return
    else:
        mydb = get_database_connection()
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM my_table")
        myresult = mycursor.fetchall()
        user_id = len(myresult) + 1  # Auto-generate user ID

        sql = "INSERT INTO my_table(Id, Name, Roll) VALUES(%s, %s, %s)"
        val = (user_id, t1.get(), t2.get())
        mycursor.execute(sql, val)
        mydb.commit()

        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                return None
            for (x, y, w, h) in faces:
                return img[y:y+h, x:x+w]

        cap = cv2.VideoCapture(0)
        img_id = 0

        while True:
            ret, frame = cap.read()
            if face_cropped(frame) is not None:
                img_id += 1
                face = cv2.resize(face_cropped(frame), (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = f"data/user.{user_id}.{img_id}.jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Cropped Face", face)
            if cv2.waitKey(1) == 13 or img_id == 50:  # Press 'Enter' or collect 50 samples
                break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result', 'Dataset generation completed!')

# Function to train the classifier
def train_classifier():
    data_dir = "data"
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        image_np = np.array(img, 'uint8')
        user_id = int(os.path.split(image)[1].split(".")[1])
        faces.append(image_np)
        ids.append(user_id)

    ids = np.array(ids)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    messagebox.showinfo('Result', 'Training completed!')

# Function to log attendance into MySQL
def log_attendance(user_id, name, roll):
    mydb = get_database_connection()
    mycursor = mydb.cursor()
    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    date = now.strftime("%Y-%m-%d")
    mycursor.execute("SELECT * FROM attendance_table WHERE UserId = %s AND Date = %s", (user_id, date))
    existing_attendance = mycursor.fetchone()

    if existing_attendance:
        # If attendance already exists, show message and do not log again
        messagebox.showinfo('Result', f'{name} has already been marked for attendance today!')
    else:
        # If no attendance found, log the new attendance
        sql = "INSERT INTO attendance_table(UserId, Name, Date, Time, Roll) VALUES (%s, %s, %s, %s, %s)"
        val = (user_id, name, date, time, roll)
        mycursor.execute(sql, val)
        mydb.commit()
        messagebox.showinfo('Result', f'Attendance for {name} has been logged!')   

# Function to detect faces and log attendance
def recognize_faces():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    mydb = get_database_connection()
    mycursor = mydb.cursor()

    cap = cv2.VideoCapture(0)
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=3)  # Stop the camera after 3 seconds

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        recognized = False  # Flag to track if any face was recognized

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            user_id, confidence = clf.predict(roi_gray)
            confidence = int(100 * (1 - confidence / 300))

            mycursor.execute("SELECT Name, Roll FROM my_table WHERE Id=%s", (user_id,))
            result = mycursor.fetchone()

            if result and confidence > 75:  # If face is recognized with enough confidence
                name, roll = result
                cv2.putText(frame, f"{name}{confidence}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                log_attendance(user_id, name, roll)
                recognized = True
                break  # Exit the loop once a recognized face is detected
            else:
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Face Recognition", frame)

        # If face is recognized, stop the camera after 3 seconds
        if recognized and datetime.now() > end_time:
            break

        # If unknown face detected, continue running the camera
        if cv2.waitKey(1) == 13:  # Press 'Enter' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Buttons for the application
b1 = tk.Button(window, text="Generate Dataset", font=("Arial", 20), bg="pink", fg="black", command=generate_dataset)
b1.grid(column=0, row=4)

b2 = tk.Button(window, text="Train Classifier", font=("Arial", 20), bg="orange", fg="red", command=train_classifier)
b2.grid(column=1, row=4)

b3 = tk.Button(window, text="Log Attendance", font=("Arial", 20), bg="green", fg="white", command=recognize_faces)
b3.grid(column=2, row=4)

# Configure the window
window.geometry("800x300")
window.mainloop()


