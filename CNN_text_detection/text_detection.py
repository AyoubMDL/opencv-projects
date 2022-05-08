import numpy as np
from scripts.utils import *
from tensorflow.keras.models import load_model
from scripts.cnn_training import preprocess_image

frame_width = 720
frame_height = 480
brightness = 10

model = load_model("./model.model")

if __name__ == "__main__":
    previous_time = 0
    cap = cv2.VideoCapture(0)
    cap.set(3, frame_width)
    cap.set(4, frame_height)
    cap.set(10, brightness)

    while True:
        success, original = cap.read()
        img = np.asarray(original)
        img = cv2.resize(img, (32, 32))
        img = preprocess_image(img)
        img = np.reshape(img, (1, 32, 32, 1))
        result = model.predict(img)
        label = np.argmax(result, axis=1)[0]
        probability = result[0][label]
        if probability > 0.65:
            cv2.putText(original, str(label) + " " + str(probability), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 0), 2)

        cv2.putText(original, f'FPS : {int(compute_fps(previous_time))}', (450, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0),
                    2)
        previous_time = time.time()

        cv2.imshow("predict number", original)
        if quit_program("q"):
            break
