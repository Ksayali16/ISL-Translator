from flask import Flask, Response, jsonify, request
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from flask_cors import CORS
import string
import time
import json

app = Flask(__name__)
CORS(app)

# =======================
# Load models
# =======================
model1 = load_model('numericalhand_gesture_model2.0.h5')
model2 = load_model('onehand_gesture_model.h5')
model3 = load_model('twohand_gesture_model.h5')

num_label_mapping = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9"
}

image_model = load_model('image_hand_gesture_model2.0.h5')
class_indices = np.load('class_mapping.npy', allow_pickle=True).item()
class_mapping = {v: k for k, v in class_indices.items()}

cap = cv2.VideoCapture(0)
last_predictions = {"model1": "", "model2": "", "model3": "", "sign": ""}
sign_state = {"current_word": "", "mode": "alphabet", "running": True}

pending_prediction = {"label": "", "start_time": 0}
CONFIRMATION_DELAY = 3  # seconds

# =======================
# Alphabet Image Mapping
# =======================
BASE_PATH = r"C:\Users\Admin\Documents\Btechproject\Indian"
alphabet_dict = {}
for ch in string.ascii_uppercase:
    path = os.path.join(BASE_PATH, ch, "4.jpg")
    if os.path.exists(path):
        alphabet_dict[ch] = path
for digit in string.digits:
    path = os.path.join(BASE_PATH, digit, "4.jpg")
    if os.path.exists(path):
        alphabet_dict[digit] = path

# =======================
# Label mappings
# =======================
onehand_classes = os.listdir('onehand_landmarks') if os.path.exists('onehand_landmarks') else []
onehand_label_mapping = {i: class_name for i, class_name in enumerate(onehand_classes)}

class_labels = np.load('y.npy') if os.path.exists('y.npy') else np.array([])
unique_twohand_labels = np.unique(class_labels) if class_labels.size else np.array([])
twohand_label_mapping = {i: label for i, label in enumerate(unique_twohand_labels)}

# =======================
# Word suggestion setup
# =======================
word_list = []
try:
    import nltk
    from nltk.corpus import words
    try:
        word_list = [w.lower() for w in words.words()]
    except Exception:
        nltk.download('words', quiet=True)
        word_list = [w.lower() for w in words.words()]
except Exception:
    fallback_path = os.path.join(os.getcwd(), 'words.txt')
    if os.path.exists(fallback_path):
        with open(fallback_path, 'r', encoding='utf-8', errors='ignore') as f:
            word_list = [w.strip().lower() for w in f if w.strip()]

def suggest_words(prefix, max_suggestions=6):
    if not prefix or not word_list:
        return []
    p = prefix.lower()
    suggestions = [w for w in word_list if w.startswith(p)]
    return suggestions[:max_suggestions]

# =======================
# Sign-to-Text Stream
# =======================
def gen_sign_to_text():
    global cap, last_predictions, sign_state, model1, model2, model3, pending_prediction
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5,
                           min_tracking_confidence=0.5,
                           max_num_hands=2)
    window_name = "SignToTextControl"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 300, 200)

    while sign_state.get("running", True):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            sign_state["running"] = False
            break

        model = None
        label = ""
        try:
            if sign_state["mode"] == "numeric":
                model = model1
            else:
                if num_hands == 1:
                    model = model2
                elif num_hands == 2:
                    model = model3

            if results.multi_hand_landmarks and model is not None:
                if sign_state["mode"] == "numeric":
                    hand0 = results.multi_hand_landmarks[0]
                    lm_list = []
                    for lm in hand0.landmark:
                        lm_list.extend([lm.x, lm.y])
                    if lm_list:
                        pred = model.predict(np.array([lm_list]))
                        idx = int(np.argmax(pred))
                        label = num_label_mapping.get(idx, str(idx))
                        last_predictions["model1"] = label
                else:
                    if num_hands == 1:
                        hand0 = results.multi_hand_landmarks[0]
                        lm_list = []
                        for lm in hand0.landmark:
                            lm_list.extend([lm.x, lm.y])
                        if lm_list:
                            pred = model.predict(np.array([lm_list]))
                            idx = int(np.argmax(pred))
                            label = onehand_label_mapping.get(idx, str(idx))
                            last_predictions["model2"] = label
                    elif num_hands == 2:
                        lm_all = []
                        for hand_landmarks in results.multi_hand_landmarks:
                            for lm in hand_landmarks.landmark:
                                lm_all.extend([lm.x, lm.y])
                        if lm_all:
                            pred = model.predict(np.array([lm_all]))
                            idx = int(np.argmax(pred))
                            label = str(twohand_label_mapping.get(idx, str(idx)))
                            last_predictions["model3"] = label
        except Exception as ex:
            print("Prediction error (ignore if intermittent):", ex)
            label = ""

        # Confirmation logic
        if label:
            now = time.time()
            cv2.putText(frame, f"Pred: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if label != pending_prediction["label"]:
                pending_prediction["label"] = label
                pending_prediction["start_time"] = now

            elapsed = now - pending_prediction["start_time"]
            remaining = max(0, CONFIRMATION_DELAY - elapsed)
            color = (0, 255, 255) if remaining > 0 else (0, 255, 0)
            cv2.putText(frame, f"Confirming in: {remaining:.1f}s", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if elapsed >= CONFIRMATION_DELAY:
                if sign_state["mode"] == "alphabet":
                    ch = label.strip()
                    if ch and ch[0].isalpha():
                        sign_state["current_word"] += ch[0].lower()
                        sign_state["current_word"] = sign_state["current_word"][-30:]
                        last_predictions["sign"] = sign_state["current_word"]
                else:
                    digit = label.strip()
                    if digit.isdigit():
                        sign_state["current_word"] += digit
                        sign_state["current_word"] = sign_state["current_word"][-30:]
                        last_predictions["sign"] = sign_state["current_word"]

                pending_prediction["label"] = ""
                pending_prediction["start_time"] = 0

        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Mode: {sign_state['mode']}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Text: {sign_state['current_word']}", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(window_name, np.zeros((200, 300, 3), dtype=np.uint8))
        ret2, buffer = cv2.imencode('.jpg', frame)
        if not ret2:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

    try:
        cv2.destroyWindow(window_name)
    except Exception:
        pass

# =======================
# Flask Routes
# =======================
@app.route('/video_feed')
def video_feed():
    sign_state["current_word"] = ""
    sign_state["mode"] = "alphabet"
    sign_state["running"] = True
    return Response(gen_sign_to_text(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/suggestions')
def suggestions_route():
    prefix = request.args.get('prefix', '').strip().lower()
    suggestions = suggest_words(prefix, max_suggestions=6)
    return jsonify({"suggestions": suggestions})

@app.route('/last-prediction/<which>')
def get_last_prediction(which):
    if which in last_predictions:
        return jsonify({"prediction": last_predictions[which]})
    return jsonify({"error": "Invalid model"}), 400

@app.route('/set-mode/<mode>')
def set_mode(mode):
    if mode in ["alphabet", "numeric"]:
        sign_state["mode"] = mode
        return jsonify({"status": "ok", "mode": mode})
    return jsonify({"error": "invalid mode"}), 400

def predict_image(img_path, target_size=(64, 64)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = image_model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_index])
    predicted_class = class_mapping.get(predicted_class_index, "Unknown")
    return predicted_class, confidence

@app.route('/predict-image', methods=['POST'])
def predict_image_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    temp_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(temp_path)
    try:
        predicted_class, confidence = predict_image(temp_path)
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence
        })
    finally:
        os.remove(temp_path)

@app.route('/text-to-sign/<text>')
def text_to_sign_feed(text):
    text = text.strip()
    words = text.split()
    line_images = []
    max_width = 0
    for word in words:
        images = []
        for char in word:
            if char.isalpha() or char.isdigit():
                img_path = alphabet_dict.get(char.upper())
                if img_path and os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    images.append(img)
        if images:
            word_image = np.hstack(images)
            line_images.append(word_image)
            max_width = max(max_width, word_image.shape[1])
    if not line_images:
        return "No valid characters", 400
    padded_line_images = []
    for line_image in line_images:
        padded_image = np.zeros((line_image.shape[0], max_width, 3), dtype=np.uint8)
        padded_image[:, :line_image.shape[1], :] = line_image
        padded_line_images.append(padded_image)
    combined_image = np.vstack(padded_line_images)
    _, buffer = cv2.imencode('.jpg', combined_image)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

# =======================
# HTML Page
# =======================
html_code = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Sign-to-Text</title>
<style>
 body{font-family: Arial, sans-serif;margin:0;padding:0}
 .container{display:flex;height:100vh}
 .left{width:20%;background:#f6f6f6;padding:20px;border-right:1px solid #ddd}
 .center{width:60%;padding:10px;display:flex;align-items:center;justify-content:center;background:#222}
 .right{width:20%;padding:20px;background:#f6f6f6;border-left:1px solid #ddd}
 button{width:100%;padding:10px;margin-bottom:10px;border-radius:6px;border:none;background:#2b8be6;color:white;cursor:pointer}
 button.secondary{background:#4CAF50}
 img#stream{width:100%;height:calc(100vh - 40px);object-fit:contain;background:black}
 textarea{width:100%;height:120px;border-radius:6px;padding:10px;border:1px solid #ccc;resize:none}
 .suggestions{margin-top:10px}
 .suggestion{display:inline-block;padding:6px 10px;margin:6px;background:#eee;border-radius:4px;cursor:pointer}
</style>
</head>
<body>
<div class="container">
  <div class="left">
    <button onclick="startSign()">SIGN TO TEXT</button>
    <div id="modeButtons" style="display:none">
      <button onclick="setMode('alphabet')">ALPHABET MODE</button>
      <button onclick="setMode('numeric')">NUMERIC MODE</button>
    </div>
    <button onclick="triggerUpload()" class="secondary">UPLOAD IMAGE</button>
    <input id="imginput" type="file" accept="image/*" style="display:none" />
  </div>
  <div class="center">
    <img id="stream" src="" alt="Stream" />
  </div>
  <div class="right">
    <label>Predictions / Prompt:</label>
    <textarea id="prompt" placeholder="Predictions will append here"></textarea>
    <button onclick="convertText()">CONVERT</button>
    <div class="suggestions" id="suggestions"></div>
    <div style="font-size:12px;margin-top:10px;color:#666">
      Tip: Focus camera, press 'a' for alphabet mode, 'n' for numeric mode. Press ESC in the small OpenCV window to stop.
    </div>
  </div>
</div>

<script>
let streamActive = false;
let pollTimer = null;

function startSign(){
    document.getElementById('stream').src = '/video_feed';
    document.getElementById('modeButtons').style.display = 'block';
    streamActive = true;
    if(pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(pollPredictions, 400);
}

function pollPredictions(){
    fetch('/last-prediction/sign').then(r=>r.json()).then(data=>{
        if(data.prediction !== undefined){
            const prompt = document.getElementById('prompt');
            if(prompt.value !== data.prediction){
                prompt.value = data.prediction;
            }
            let prefix = data.prediction.split(' ').pop();
            if(prefix === undefined) prefix = data.prediction;
            fetch('/suggestions?prefix=' + encodeURIComponent(prefix))
              .then(r=>r.json()).then(js=>{
                renderSuggestions(js.suggestions || []);
              });
        }
    }).catch(err=>{});
}

function renderSuggestions(list){
    const box = document.getElementById('suggestions');
    box.innerHTML = '';
    list.forEach(word=>{
        const el = document.createElement('span');
        el.className = 'suggestion';
        el.textContent = word;
        el.onclick = ()=> {
            const prompt = document.getElementById('prompt');
            let tokens = prompt.value.split(' ');
            tokens[tokens.length - 1] = word;
            prompt.value = tokens.join(' ');
        };
        box.appendChild(el);
    });
}

function triggerUpload(){
    const input = document.getElementById('imginput');
    input.onchange = ()=>{
        const file = input.files[0];
        if(!file) return;
        const fd = new FormData();
        fd.append('file', file);
        fetch('/predict-image', { method: 'POST', body: fd })
        .then(r=>r.json()).then(js=>{
            const prompt = document.getElementById('prompt');
            prompt.value += js.predicted_class + ' ';
        }).catch(e=>alert('Upload failed'));
    };
    input.click();
}

function setMode(mode){
    fetch('/set-mode/' + mode)
      .then(r=>r.json())
      .then(js=>{
          alert("Switched to " + js.mode + " mode");
      });
}

function convertText(){
    const text = document.getElementById('prompt').value.trim();
    if(text) {
        document.getElementById('stream').src = '/text-to-sign/' + encodeURIComponent(text);
    }
}

window.addEventListener('beforeunload', ()=>{
    if(pollTimer) clearInterval(pollTimer);
});
</script>
</body>
</html>"""

@app.route('/')
def index():
    return html_code

@app.route('/last-prediction/sign')
def last_sign_prediction():
    return jsonify({"prediction": sign_state.get("current_word", "")})

if __name__ == '__main__':
    print("Running on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
