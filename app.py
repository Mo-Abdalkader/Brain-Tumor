from flask import Flask, render_template, request, jsonify, Response, session
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import uuid
from datetime import datetime
import time
from flask_httpauth import HTTPBasicAuth
import io
import base64

app = Flask(__name__, static_folder='static', template_folder='templates')
auth = HTTPBasicAuth()

# ========== Configuration ==========
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
STATS_FILE = 'visitor_stats.json'
ACTIVE_CONNECTIONS = set()  # For real-time tracking

app.config.update({
    'UPLOAD_FOLDER': UPLOAD_FOLDER,
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB
    'SECRET_KEY': "MOHAMED"  # Change for production
})

app.secret_key = app.config['SECRET_KEY']

# ========== Authentication ==========
users = {
    "admin": generate_password_hash("Mo_Abdalkader")
}


@auth.verify_password
def verify_password(username, password):
    if username in users:
        return check_password_hash(users.get(username), password)
    return False


@auth.error_handler
def auth_error(status):
    return jsonify({"error": "Access denied"}), status


# ========== Visitor Tracking ==========
def init_stats():
    """Initialize stats file with default structure"""
    default_stats = {
        "total_visits": 0,
        "unique_visitors": 0,
        "ip_data": {},
        "timestamps": {}
    }

    if not os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'w') as f:
            json.dump(default_stats, f)
    else:
        with open(STATS_FILE, 'r+') as f:
            try:
                stats = json.load(f)
                for key in default_stats:
                    if key not in stats:
                        stats[key] = default_stats[key]
                f.seek(0)
                json.dump(stats, f, indent=4)
                f.truncate()
            except (json.JSONDecodeError, IOError):
                with open(STATS_FILE, 'w') as f:
                    json.dump(default_stats, f)


def update_stats(ip):
    """Update visitor statistics"""
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'r') as f:
                stats = json.load(f)
        else:
            stats = {
                "total_visits": 0,
                "unique_visitors": 0,
                "ip_data": {},
                "timestamps": {}
            }

        stats['ip_data'] = stats.get('ip_data', {})
        stats['timestamps'] = stats.get('timestamps', {})

        stats['total_visits'] = stats.get('total_visits', 0) + 1

        if 'visitor_id' not in session:
            session['visitor_id'] = str(uuid.uuid4())
            stats['unique_visitors'] = stats.get('unique_visitors', 0) + 1

        stats['ip_data'][ip] = stats['ip_data'].get(ip, 0) + 1
        stats['timestamps'][ip] = datetime.now().isoformat()

        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=4)

        return stats
    except Exception as e:
        print(f"Error updating stats: {e}")
        init_stats()
        return update_stats(ip)


# ========== Model Setup ==========
try:
    model = load_model('models/final_model.keras')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

CLASS_LABELS = {
    0: 'Glioma',
    1: 'Meningioma',
    3: 'Pituitary',
    2: 'No Tumor'
}

DISEASE_INFO = {
    'Glioma': {
        'description': 'A tumor that occurs in the brain and spinal cord, originating from glial cells.',
        'location': 'Brain and spinal cord glial cells',
        'danger': 'High (can be malignant)',
        'advice': 'Consult a neuro-oncologist immediately. Treatment may include surgery, radiation, and chemotherapy.'
    },
    'Meningioma': {
        'description': 'A tumor that arises from the meninges, the membranes surrounding the brain and spinal cord.',
        'location': 'Meninges (brain and spinal cord membranes)',
        'danger': 'Medium (usually benign but can grow large)',
        'advice': 'Consult a neurologist. Treatment often involves monitoring or surgery depending on size and symptoms.'
    },
    'Pituitary': {
        'description': 'A tumor that forms in the pituitary gland, affecting hormone production.',
        'location': 'Pituitary gland (base of the brain)',
        'danger': 'Low to Medium (typically benign)',
        'advice': 'Consult an endocrinologist. Treatment may include medication or surgery to restore hormone balance.'
    },
    'No Tumor': {
        'description': 'No signs of tumorous growth detected in the brain scan.',
        'location': 'N/A',
        'danger': 'None',
        'advice': 'Continue regular check-ups. Maintain a healthy lifestyle with proper diet and exercise.'
    }
}


# ========== Core Routes ==========
@app.before_request
def track_visitor():
    """Track all visitors before processing requests"""
    init_stats()
    ip = request.remote_addr
    ACTIVE_CONNECTIONS.add(ip)
    update_stats(ip)


@app.teardown_request
def remove_connection(exception=None):
    """Remove disconnected clients"""
    ip = request.remote_addr
    if ip in ACTIVE_CONNECTIONS:
        ACTIVE_CONNECTIONS.remove(ip)


@app.route('/')
def home():
    """Main application page"""
    try:
        with open(STATS_FILE) as f:
            stats = json.load(f)
        return render_template('index.html',
                               total_visits=stats.get('total_visits', 0),
                               unique_visitors=stats.get('unique_visitors', 0))
    except Exception as e:
        print(f"Error loading stats: {e}")
        return render_template('index.html', total_visits=0, unique_visitors=0)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image predictions"""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            # Read the file directly to memory
            img_bytes = file.read()

            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_bytes))

            # Save to temporary file for processing
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            img.save(temp_path)

            # Process and predict
            processed_img = preprocess_image(temp_path)
            predictions = model.predict(processed_img)
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            class_name = CLASS_LABELS.get(predicted_class, 'Unknown')

            # Convert image to base64 for display
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return jsonify({
                'class': class_name,
                'confidence': confidence,
                'image_data': img_str,
                'info': DISEASE_INFO.get(class_name, {})
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass

    return jsonify({'error': 'File type not allowed'}), 400


# ========== Statistics Routes ==========
@app.route('/get_stats')
def get_stats():
    """JSON endpoint for visitor stats"""
    try:
        with open(STATS_FILE) as f:
            return jsonify(json.load(f))
    except Exception as e:
        print(f"Error loading stats: {e}")
        return jsonify({"error": "Could not load statistics"}), 500


@app.route('/stats')
@auth.login_required
def stats_dashboard():
    """Admin statistics dashboard"""
    try:
        with open(STATS_FILE) as f:
            stats = json.load(f)
        return render_template('stats.html',
                               total_visits=stats.get('total_visits', 0),
                               unique_visitors=stats.get('unique_visitors', 0),
                               ip_data=stats.get('ip_data', {}),
                               online_now=len(ACTIVE_CONNECTIONS))
    except Exception as e:
        print(f"Error loading stats: {e}")
        return render_template('stats.html',
                               total_visits=0,
                               unique_visitors=0,
                               ip_data={},
                               online_now=0)


@app.route('/live_stats')
def live_stats():
    """Server-Sent Events for real-time updates"""

    def event_stream():
        while True:
            try:
                with open(STATS_FILE) as f:
                    stats = json.load(f)
                data = {
                    "current_online": len(ACTIVE_CONNECTIONS),
                    "total_visits": stats.get('total_visits', 0),
                    "unique_visitors": stats.get('unique_visitors', 0)
                }
                yield f"data: {json.dumps(data)}\n\n"
            except Exception as e:
                print(f"Error in live stats: {e}")
                yield f"data: {json.dumps({'error': 'stats unavailable'})}\n\n"
            time.sleep(10)

    return Response(event_stream(), mimetype="text/event-stream")


# ========== Helper Functions ==========
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size, color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


# ========== Startup ==========
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    init_stats()
    app.run(port=5000, debug=False)
