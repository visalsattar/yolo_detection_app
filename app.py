from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Updated to a stable, relative path
MODEL_PATH = os.path.join(APP_ROOT, 'models', 'best.pt')
model = YOLO(MODEL_PATH)

# Extracted to global scope to avoid redefining on every single prediction
CLASS_NAMES = [
    'animal_crab', 'animal_eel', 'animal_etc', 'animal_fish', 'animal_shells',
    'animal_starfish', 'plant', 'rov', 'trash_bag', 'trash_bottle', 'trash_branch',
    'trash_can', 'trash_clothing', 'trash_container', 'trash_cup', 'trash_net',
    'trash_pipe', 'trash_rope', 'trash_snack_wrapper', 'trash_tarp', 'trash_unknown_instance',
    'trash_wreckage'
]

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        allowed = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        if file and os.path.splitext(file.filename)[1].lower() in allowed:
            # secure_filename strips path components such as ../ so uploads can't escape static/images
            filename = secure_filename(file.filename)
            path = os.path.join(APP_ROOT, 'static', 'images', filename)
            file.save(path)
            return predict(path)
    return render_template('upload.html')

def predict(image_path):
    img = cv2.imread(image_path)
    results = model(img)

    for result in results:
        for *xyxy, conf, cls in result.boxes.data:
            label = f'{CLASS_NAMES[int(cls)]} {conf:.2f}'
            color = (255, 0, 0)
            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            cv2.rectangle(img, c1, c2, color, thickness=2)
            cv2.putText(img, label, (c1[0], c1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_filename = os.path.basename(image_path)
    output_path = os.path.join(APP_ROOT, 'static', 'predicted', output_filename)
    cv2.imwrite(output_path, img)
    
    return render_template('result.html', user_image=output_filename)

if __name__ == '__main__':
    # Debug mode exposes the Werkzeug debugger; keep it off unless developing locally
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1')