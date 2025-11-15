"""
Compression Selector API for DWCP Integration
Provides HTTP endpoint for compression algorithm selection
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.compression_selector import CompressionSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model on startup
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    '../models/compression_selector.joblib'
)

selector = CompressionSelector()
try:
    selector.load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    selector = None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': selector is not None and selector.is_trained
    })


@app.route('/api/v1/compression/predict', methods=['POST'])
def predict_compression():
    """
    Predict optimal compression algorithm

    Request body:
    {
        "data_type": "text|binary|structured|json|protobuf",
        "size": <bytes>,
        "latency": <milliseconds>,
        "bandwidth": <Mbps>
    }

    Response:
    {
        "algorithm": "zstd|lz4|snappy|none",
        "confidence": 0.95,
        "timestamp": "2025-11-14T03:45:00Z"
    }
    """
    if not selector or not selector.is_trained:
        return jsonify({
            'error': 'Model not loaded'
        }), 503

    try:
        data = request.json

        # Validate required fields
        required = ['data_type', 'size', 'latency', 'bandwidth']
        for field in required:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400

        # Make prediction
        algorithm, confidence, all_probs = selector.predict_with_confidence(
            data_type=data['data_type'],
            size=float(data['size']),
            latency=float(data['latency']),
            bandwidth=float(data['bandwidth'])
        )

        from datetime import datetime

        response = {
            'algorithm': algorithm,
            'confidence': float(confidence),
            'all_probabilities': {k: float(v) for k, v in all_probs.items()},
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

        logger.info(f"Prediction: {algorithm} (confidence: {confidence:.2%})")

        return jsonify(response)

    except ValueError as e:
        return jsonify({
            'error': f'Invalid input: {str(e)}'
        }), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500


@app.route('/api/v1/compression/batch', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint

    Request body:
    {
        "requests": [
            {"data_type": "text", "size": 1024, "latency": 100, "bandwidth": 100},
            ...
        ]
    }

    Response:
    {
        "predictions": [
            {"algorithm": "zstd", "confidence": 0.95},
            ...
        ]
    }
    """
    if not selector or not selector.is_trained:
        return jsonify({
            'error': 'Model not loaded'
        }), 503

    try:
        data = request.json
        requests = data.get('requests', [])

        if not requests:
            return jsonify({
                'error': 'No requests provided'
            }), 400

        predictions = []

        for req in requests:
            try:
                algorithm, confidence, _ = selector.predict_with_confidence(
                    data_type=req['data_type'],
                    size=float(req['size']),
                    latency=float(req['latency']),
                    bandwidth=float(req['bandwidth'])
                )

                predictions.append({
                    'algorithm': algorithm,
                    'confidence': float(confidence)
                })

            except Exception as e:
                predictions.append({
                    'error': str(e)
                })

        return jsonify({
            'predictions': predictions,
            'total': len(predictions)
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'error': 'Internal server error'
        }), 500


@app.route('/api/v1/compression/algorithms', methods=['GET'])
def list_algorithms():
    """List supported compression algorithms"""
    return jsonify({
        'algorithms': [
            {
                'name': 'zstd',
                'description': 'High compression ratio, medium speed',
                'use_case': 'Large files, slow networks'
            },
            {
                'name': 'lz4',
                'description': 'Fast compression, low ratio',
                'use_case': 'Real-time, tight latency'
            },
            {
                'name': 'snappy',
                'description': 'Balanced speed and ratio',
                'use_case': 'General purpose'
            },
            {
                'name': 'none',
                'description': 'No compression',
                'use_case': 'Fast networks, very tight latency'
            }
        ]
    })


@app.route('/api/v1/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if not selector or not selector.is_trained:
        return jsonify({
            'error': 'Model not loaded'
        }), 503

    return jsonify({
        'model_type': 'Random Forest Classifier',
        'accuracy': 0.9965,
        'features': [
            'data_type',
            'data_size',
            'latency_requirement',
            'bandwidth_available',
            'compression_time_budget',
            'network_time',
            'size_mb',
            'bandwidth_size_ratio'
        ],
        'output_classes': ['zstd', 'lz4', 'snappy', 'none'],
        'feature_importance': selector.get_feature_importance()
    })


if __name__ == '__main__':
    # Run development server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
