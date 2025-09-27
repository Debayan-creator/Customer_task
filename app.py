
# Save this as app.py for deployment
from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import numpy as np
import time

app = Flask(__name__)

# Load model and encoder
model = tf.keras.models.load_model('ticket_triage_model')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

@app.route('/classify', methods=['POST'])
def classify_ticket():
    try:
        data = request.json
        subject = data.get('subject', '')
        description = data.get('description', '')
        
        # Combine text
        text = subject + ' ' + description
        
        # Predict
        start_time = time.time()
        predictions = model.predict([text], verbose=0)
        latency = (time.time() - start_time) * 1000
        
        # Get results
        predicted_idx = np.argmax(predictions[0])
        predicted_category = label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(predictions[0][predicted_idx])
        
        return jsonify({
            'success': True,
            'category': predicted_category,
            'confidence': confidence,
            'latency_ms': latency
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
