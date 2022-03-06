import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from pygame import mixer

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
model = tf.keras.models.load_model('my_model.h5')

cap = cv2.VideoCapture(1)

mixer.init()
sound = mixer.Sound(r'alarm.wav')

# Check webcam is open correctly or not
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
else:
    raise IOError("Can't open webcam")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

    # eyes_roi = None
    # for (x, y, w, h) in eyes:
    #     roi_gray = gray[y:y + h, x:x + w]
    #     roi_color = frame[y:y + h, x:x + w]
    #     eyess = eye_cascade.detectMultiScale(roi_gray)
    #     if len(eyess) == 0:
    #         print("eyes not detected")
    #     else:
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for ex, ey, ew, eh in eyes:
        # eyes_roi = roi_color[ey:ey + eh, ex:ex + ew]
        eyes_roi = frame[ey:ey + eh, ex:ex + ew]
        cv2.rectangle(frame, pt1=(ex, ey), pt2=(ex + ew, ey + eh), color=(255, 0, 0), thickness=3)

        final_image = cv2.resize(eyes_roi, (224,224))
        final_image = final_image / 255.0
        final_image = final_image.reshape(224, 224, 3)
        final_image = np.expand_dims(final_image, axis=0)

        prediction = model.predict(final_image)

        if prediction > 0:
            status = "Open eyes"
            cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_4)
        elif prediction < 0:
            status = "Closed eyes"
            cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_4)
            sound.play()

    cv2.imshow(" Drowsiness Deection", frame)
    plt.show()

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
