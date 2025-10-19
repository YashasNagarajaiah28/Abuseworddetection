from flask import Flask, render_template, request, redirect, url_for
import os, re, unicodedata
import cv2
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# text cleaning
def normalize_text(t):
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[@$€0]+", "o", t)
    t = re.sub(r"[_\-\.\|/\\]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# Load NLP model once
MODEL_NAME = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
clf = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

@app.route('/')
def index():
    return render_template('index.html')

# IMAGE PROCESSING
@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files['image']
    if not file:
        return redirect(url_for('index'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Loads image
    img = Image.open(filepath).convert("RGB")
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    
    yolo_model = YOLO('yolov8n.pt')
    results = yolo_model(cv_img)
    detections = results[0].boxes.data.cpu().numpy()

    car_boxes = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 2:  # only car class
            car_boxes.append((int(x1), int(y1), int(x2), int(y2)))

    reader = easyocr.Reader(['en'], gpu=False)
    texts = []

    for (x1, y1, x2, y2) in car_boxes:
        crop = cv_img[y1:y2, x1:x2]
        ocr_result = reader.readtext(crop, detail=1, paragraph=True)
        text = " ".join([t[1] for t in ocr_result if len(t) >= 2])
        clean_text = normalize_text(text)
        if clean_text:
            texts.append(clean_text)
    
    combined_text = " ".join(texts) if texts else "No text detected"
    return render_template('text_display.html', mode="image", text=combined_text, file=file.filename)

# VOICE PROCESSING
@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    file = request.files['audio']
    if not file:
        return redirect(url_for('index'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Speech-to-text
    r = sr.Recognizer()
    with sr.AudioFile(filepath) as source:
        audio_data = r.record(source)
        try:
            voice_text = r.recognize_google(audio_data)
        except sr.UnknownValueError:
            voice_text = ""
        except sr.RequestError:
            voice_text = ""
    
    clean_text = normalize_text(voice_text)
    return render_template('text_display.html', mode="voice", text=clean_text, file=file.filename)

# TOXICITY DETECTION
@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    file = request.form['file']
    mode = request.form['mode']
    
    if not text or text == "No text detected":
        return render_template('result.html', file=file, mode=mode, text=text, label="No text found", score=0, is_abusive=False)
    
    results = clf(text)[0]
    top = max(results, key=lambda x: x['score'])
    label, score = top['label'].lower(), top['score']
    is_abusive = ("toxic" in label or "off" in label or "hate" in label) and score >= 0.6

    return render_template('result.html', file=file, mode=mode, text=text, label=label, score=round(score, 2), is_abusive=is_abusive)

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
