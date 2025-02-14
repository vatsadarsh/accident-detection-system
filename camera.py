import numpy as np
from keras.models import model_from_json
import cv2

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
count = 1

def running():
    global count
    video = cv2.VideoCapture('video_demo.mp4')
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Preprocess frame according to model input requirements
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(gray_frame, (250, 250))
        roi = roi.astype('float32') / 255.0  # Normalize pixel values

        # Reshape the input according to model's expected shape
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=3)  # Add a channel dimension

        # Make prediction
        prob = loaded_model.predict(roi)
        prob = round(prob[0][0] * 100, 2)

        if prob > 99:
            print("Accident Detected!!")
            cv2.imshow('Frame', frame)
            cv2.imwrite(f'accident{count}.jpg', frame)  # Save the frame as a JPEG image
            count += 1

        cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
        cv2.putText(frame, "Probability: " + str(prob), (20, 30), font, 1, (255, 255, 0), 2)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    running()
