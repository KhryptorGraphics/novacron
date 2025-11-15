"""
REST API for Node Reliability Predictor
Provides HTTP endpoint for reliability predictions
"""

from flask import Flask, request, jsonify
import numpy as np
from reliability_predictor import ReliabilityPredictor
import os

app = Flask(__name__)

# Global model instance
model = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'reliability_predictor.weights.h5')


def load_model():
    """Load the trained model"""
    global model
    if model is None:
        model = ReliabilityPredictor(state_size=4, learning_rate=0.001)
        if os.path.exists(MODEL_PATH):
            model.load_model(MODEL_PATH)
            print(f"✅ Model loaded from {MODEL_PATH}")
        else:
            print(f"⚠️ Model not found at {MODEL_PATH}. Using untrained model.")
    return model


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict node reliability

    Expected JSON:
    {
        "uptime": 95.5,
        "failure_rate": 0.2,
        "network_quality": 0.85,
        "distance": 500
    }

    Response:
    {
        "reliability": 0.8742,
        "confidence": "high"
    }
    """
    try:
        data = request.get_json()

        # Validate input
        required_fields = ['uptime', 'failure_rate', 'network_quality', 'distance']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Load model
        predictor = load_model()

        # Make prediction
        reliability = predictor.predict_reliability(
            uptime=float(data['uptime']),
            failure_rate=float(data['failure_rate']),
            network_quality=float(data['network_quality']),
            distance=float(data['distance'])
        )

        # Determine confidence
        if reliability >= 0.8:
            confidence = 'high'
        elif reliability >= 0.6:
            confidence = 'medium'
        else:
            confidence = 'low'

        return jsonify({
            'reliability': float(reliability),
            'confidence': confidence,
            'recommendation': 'use' if reliability >= 0.7 else 'skip'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Predict reliability for multiple nodes

    Expected JSON:
    {
        "nodes": [
            {
                "uptime": 95.5,
                "failure_rate": 0.2,
                "network_quality": 0.85,
                "distance": 500
            },
            ...
        ]
    }

    Response:
    {
        "predictions": [
            {
                "reliability": 0.8742,
                "confidence": "high"
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()

        if 'nodes' not in data or not isinstance(data['nodes'], list):
            return jsonify({'error': 'Missing or invalid nodes array'}), 400

        # Load model
        predictor = load_model()

        # Prepare batch
        states = []
        for node in data['nodes']:
            state = predictor._normalize_state(
                uptime=float(node['uptime']),
                failure_rate=float(node['failure_rate']),
                network_quality=float(node['network_quality']),
                distance=float(node['distance'])
            )
            states.append(state[0])

        states = np.array(states)

        # Predict batch
        reliabilities = predictor.predict_batch(states)

        # Format results
        predictions = []
        for reliability in reliabilities:
            if reliability >= 0.8:
                confidence = 'high'
            elif reliability >= 0.6:
                confidence = 'medium'
            else:
                confidence = 'low'

            predictions.append({
                'reliability': float(reliability),
                'confidence': confidence,
                'recommendation': 'use' if reliability >= 0.7 else 'skip'
            })

        return jsonify({'predictions': predictions}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    predictor = load_model()

    return jsonify({
        'architecture': 'DQN',
        'state_size': predictor.state_size,
        'learning_rate': predictor.learning_rate,
        'target_accuracy': '85%',
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH),
        'summary': predictor.get_summary()
    }), 200


@app.route('/model/retrain', methods=['POST'])
def retrain_model():
    """
    Retrain model with new data

    Expected JSON:
    {
        "data": [
            {
                "uptime": 95.5,
                "failure_rate": 0.2,
                "network_quality": 0.85,
                "distance": 500,
                "actual_reliability": 0.87
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()

        if 'data' not in data or not isinstance(data['data'], list):
            return jsonify({'error': 'Missing or invalid data array'}), 400

        # Load model
        predictor = load_model()

        # Prepare training data
        X = []
        y = []
        for item in data['data']:
            state = predictor._normalize_state(
                uptime=float(item['uptime']),
                failure_rate=float(item['failure_rate']),
                network_quality=float(item['network_quality']),
                distance=float(item['distance'])
            )
            X.append(state[0])
            y.append(float(item['actual_reliability']))

        X = np.array(X)
        y = np.array(y)

        # Retrain
        history = predictor.train(X, y, epochs=10, batch_size=32)

        # Save updated model
        predictor.save_model(MODEL_PATH)

        return jsonify({
            'status': 'retrained',
            'samples': len(X),
            'final_loss': float(history['loss'][-1]),
            'final_accuracy': float(history['accuracy'][-1])
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Load model on startup
    load_model()

    # Run server
    app.run(host='0.0.0.0', port=5002, debug=False)
