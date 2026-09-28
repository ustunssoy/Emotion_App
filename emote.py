"""Gerçek zamanlı duygu analizi - tek dosya, otomatik kurulum.

Çalıştırma: python app.py
İlk çalıştırmada: eksik paketleri kurar -> FER2013'ü indirir -> modeli eğitir -> sunucuyu başlatır.
"""
import importlib
import os
import subprocess
import sys

# ---------- 1) Eksik paketleri otomatik kur ----------
PACKAGES = {  # import adı: pip adı
    "flask": "flask",
    "cv2": "opencv-python-headless",
    "numpy": "numpy<2",
    "tensorflow": "tensorflow",
    "kagglehub": "kagglehub",
}
for module, pip_name in PACKAGES.items():
    try:
        importlib.import_module(module)
    except ImportError:
        print(f"[kurulum] {pip_name} kuruluyor...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

import base64  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from flask import Flask, jsonify, render_template_string, request  # noqa: E402
from tensorflow.keras.models import load_model  # noqa: E402

MODEL_PATH = "emotion_model.keras"
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTIONS = ["Kızgın", "İğrenmiş", "Korkmuş", "Mutlu", "Nötr", "Üzgün", "Şaşırmış"]

HTML = """<!DOCTYPE html>
<html lang="tr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Duygu Analizi</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;text-align:center;margin:0;padding:20px}
.stage{position:relative;display:inline-block;max-width:100%}
video,#overlay{max-width:100%;border-radius:12px}
#overlay{position:absolute;left:0;top:0;width:100%;height:100%}
#status{margin-top:12px;color:#9aa}
</style></head><body>
<h1>Gerçek Zamanlı Duygu Analizi</h1>
<div class="stage"><video id="video" autoplay playsinline muted></video><canvas id="overlay"></canvas></div>
<p id="status">Kamera bekleniyor...</p>
<script>
const video=document.getElementById("video"),overlay=document.getElementById("overlay"),
status=document.getElementById("status"),cap=document.createElement("canvas");
let busy=false;
navigator.mediaDevices.getUserMedia({video:{width:640,height:480}})
 .then(s=>{video.srcObject=s;status.textContent="Analiz ediliyor...";setInterval(tick,300)})
 .catch(e=>{status.textContent="Kamera erişimi reddedildi: "+e.message});
async function tick(){
 if(busy||!video.videoWidth)return; busy=true;
 cap.width=video.videoWidth;cap.height=video.videoHeight;
 cap.getContext("2d").drawImage(video,0,0);
 try{
  const r=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},
   body:JSON.stringify({image:cap.toDataURL("image/jpeg",0.7)})});
  const d=await r.json();
  if(d.error){status.textContent=d.error}else{draw(d);status.textContent=d.faces.length+" yüz bulundu"}
 }catch(e){status.textContent="Sunucu hatası"}
 busy=false;
}
function draw(d){
 overlay.width=d.width;overlay.height=d.height;
 const ctx=overlay.getContext("2d");ctx.clearRect(0,0,d.width,d.height);
 ctx.lineWidth=3;ctx.font="bold 22px system-ui";ctx.strokeStyle=ctx.fillStyle="#3f8";
 d.faces.forEach(f=>{const [x,y,w,h]=f.box;ctx.strokeRect(x,y,w,h);
  ctx.fillText(f.emotion+" %"+Math.round(f.score*100),x,Math.max(24,y-8))});
}
</script></body></html>"""


# ---------- 2) Veri setini indir + modeli eğit (model yoksa) ----------
def train():
    import kagglehub
    from tensorflow.keras import layers, models, utils

    print("[veri] FER2013 indiriliyor (ilk seferde birkaç dakika sürebilir)...")
    path = kagglehub.dataset_download("msambare/fer2013")
    print("[veri] Hazır:", path)

    def load(split):
        ds = utils.image_dataset_from_directory(
            os.path.join(path, split), class_names=CLASSES, color_mode="grayscale",
            image_size=(48, 48), batch_size=64, label_mode="int", shuffle=(split == "train"))
        return ds.map(lambda x, y: (x / 255.0, y)).prefetch(2)

    train_ds, val_ds = load("train"), load("test")

    m = models.Sequential([
        layers.Input((48, 48, 1)),
        layers.RandomFlip("horizontal"),
        layers.Conv2D(32, 3, activation="relu", padding="same"), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"), layers.Dropout(0.5),
        layers.Dense(7, activation="softmax"),
    ])
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    print("[eğitim] Başlıyor...")
    m.fit(train_ds, validation_data=val_ds, epochs=30)
    m.save(MODEL_PATH)
    print("[eğitim] Model kaydedildi:", MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    train()

# ---------- 3) Web uygulaması ----------
app = Flask(__name__)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model = load_model(MODEL_PATH)


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw = base64.b64decode((request.get_json(silent=True) or {})["image"].split(",")[-1])
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return jsonify(error="Geçersiz görüntü"), 400
    if frame is None:
        return jsonify(error="Görüntü çözülemedi"), 400

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5, minSize=(40, 40))
    results = []
    for (x, y, w, h) in faces:
        roi = cv2.resize(gray[y:y + h, x:x + w], (48, 48)) / 255.0
        probs = model.predict(roi.reshape(1, 48, 48, 1), verbose=0)[0]
        i = int(np.argmax(probs))
        results.append({"box": [int(x), int(y), int(w), int(h)],
                        "emotion": EMOTIONS[i], "score": round(float(probs[i]), 3)})
    return jsonify(faces=results, width=frame.shape[1], height=frame.shape[0])


if __name__ == "__main__":
    app.run(debug=False)  # debug=True eğitimi iki kez başlatabilir