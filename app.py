from flask import Flask, render_template, redirect, url_for, session, request, flash, get_flashed_messages
from flask import Response,jsonify
import cv2, sys, numpy as np, os, base64, joblib
from io import BytesIO
from PIL import Image,UnidentifiedImageError
import mediapipe as mp
import pickle
import time

app = Flask(__name__)
@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    return response

@app.route("/", methods=['GET', 'POST'])
def login():
    return render_template('index.html')

# Assuming labels_dict is a global variable
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
    6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
    12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',24: 'Y',25: 'Z',26:' '
}

@app.route('/process_video_frames', methods=['POST'])
def process_video_frames():
    global labels_dict
    global model
    
    recognized_alphabet = ''

    try:
        image_data_bytes = request.data  # Assuming request is properly defined

        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data_bytes, np.uint8)

        # Decode numpy array into an image
        image_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Check if the image is empty
        if image_cv2 is None or np.all(image_cv2 == 0):
            return jsonify({"sentence": ""})

        # Convert cv2 image to RGB  
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        model_dict = pickle.load(open('./model.p', 'rb'))
        model = model_dict['model']

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

        labels_dict = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
            6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
            12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
            18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',24: 'Y',25: 'Z',26:' '
        }


        # Initialize variables
        recognized_alphabets = []

        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []
                y_ = []
                data_aux = []  # Initialize data_aux here

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Recognize alphabet
                prediction = model.predict([np.asarray(data_aux)])
                recognized_alphabet = labels_dict[int(prediction[0])]
                recognized_alphabets.append(recognized_alphabet)
                print(recognized_alphabet)
        # Send the sentence as a response to the client in JSON format

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"recognized_alphabet": ""})
    
    return jsonify({"recognized_alphabet": recognized_alphabet})


if __name__ == '__main__':
    app.run(debug=True)