#!/usr/bin/env python3
"""
TensorFlow Model Implementation for {{experimentName}}
MLE-Star Framework - Deep Learning Model Architecture
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers, losses, metrics
import numpy as np
import logging
from pathlib import Path
import yaml
import json

logger = logging.getLogger(__name__)

# Set memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logger.warning(f"GPU configuration error: {e}")

class MLPClassifier:
    """
    Multi-Layer Perceptron Classifier using TensorFlow/Keras
    """
    
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate=0.2,
                 activation='relu', batch_norm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.batch_norm = batch_norm
        
        self.model = self._build_model()
    
    def _build_model(self):
        """Build the MLP model"""
        model = keras.Sequential([
            layers.Input(shape=(self.input_size,))
        ])
        
        # Hidden layers
        for hidden_size in self.hidden_layers:
            model.add(layers.Dense(hidden_size))
            
            if self.batch_norm:
                model.add(layers.BatchNormalization())
            
            model.add(layers.Activation(self.activation))
            
            if self.dropout_rate > 0:
                model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer
        if self.output_size == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
        else:
            model.add(layers.Dense(self.output_size, activation='softmax'))
        
        return model
    
    def compile_model(self, learning_rate=0.001, optimizer='adam'):
        """Compile the model with optimizer and loss function"""
        if optimizer == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'adamw':
            opt = optimizers.AdamW(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        if self.output_size == 2:
            loss = 'binary_crossentropy'
            metrics_list = ['accuracy', 'precision', 'recall']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics_list = ['accuracy']
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics_list
        )
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()
    
    def get_model(self):
        """Return the Keras model"""
        return self.model

class CNNClassifier:
    """
    Convolutional Neural Network for image classification
    """
    
    def __init__(self, input_shape, num_classes, architecture='simple'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.architecture = architecture
        
        self.model = self._build_model()
    
    def _build_model(self):
        """Build CNN model based on architecture"""
        if self.architecture == 'simple':
            return self._build_simple_cnn()
        elif self.architecture == 'resnet':
            return self._build_resnet()
        elif self.architecture == 'vgg':
            return self._build_vgg()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
    
    def _build_simple_cnn(self):
        """Build simple CNN architecture"""
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),
            
            # First conv block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second conv block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third conv block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            # Classification head
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax' if self.num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def _build_resnet(self):
        """Build ResNet-like architecture"""
        def residual_block(x, filters, kernel_size=3, stride=1):
            # Shortcut path
            shortcut = x
            if stride != 1:
                shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            
            # Main path
            x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            x = layers.Conv2D(filters, kernel_size, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Add shortcut
            x = layers.Add()([x, shortcut])
            x = layers.Activation('relu')(x)
            
            return x
        
        # Input
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Residual blocks
        x = residual_block(x, 32)
        x = residual_block(x, 64, stride=2)
        x = residual_block(x, 128, stride=2)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, 
                              activation='softmax' if self.num_classes > 2 else 'sigmoid')(x)
        
        return keras.Model(inputs, outputs)
    
    def _build_vgg(self):
        """Build VGG-like architecture"""
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),
            
            # Block 1
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Block 2
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Block 3
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Classification head
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, 
                        activation='softmax' if self.num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def compile_model(self, learning_rate=0.001, optimizer='adam'):
        """Compile the CNN model"""
        if optimizer == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizer
        
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
        else:
            loss = 'sparse_categorical_crossentropy'
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=['accuracy']
        )
    
    def get_model(self):
        """Return the Keras model"""
        return self.model

class LSTMClassifier:
    """
    LSTM-based classifier for sequence data
    """
    
    def __init__(self, max_sequence_length, vocab_size, embedding_dim=100, 
                 lstm_units=128, num_classes=2, dropout=0.2):
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_classes = num_classes
        self.dropout = dropout
        
        self.model = self._build_model()
    
    def _build_model(self):
        """Build LSTM model"""
        model = keras.Sequential([
            layers.Input(shape=(self.max_sequence_length,)),
            layers.Embedding(self.vocab_size, self.embedding_dim),
            layers.LSTM(self.lstm_units, dropout=self.dropout, recurrent_dropout=self.dropout,
                       return_sequences=True),
            layers.LSTM(self.lstm_units // 2, dropout=self.dropout, recurrent_dropout=self.dropout),
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.dropout),
            layers.Dense(self.num_classes, 
                        activation='softmax' if self.num_classes > 2 else 'sigmoid')
        ])
        
        return model
    
    def compile_model(self, learning_rate=0.001, optimizer='adam'):
        """Compile LSTM model"""
        if optimizer == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
        else:
            loss = 'sparse_categorical_crossentropy'
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=['accuracy']
        )
    
    def get_model(self):
        """Return the Keras model"""
        return self.model

class TransformerClassifier:
    """
    Transformer-based classifier
    """
    
    def __init__(self, max_sequence_length, vocab_size, embedding_dim=128,
                 num_heads=4, ff_dim=128, num_transformer_blocks=2, 
                 num_classes=2, dropout=0.1):
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.num_classes = num_classes
        self.dropout = dropout
        
        self.model = self._build_model()
    
    def _transformer_block(self, inputs):
        """Single transformer block"""
        # Multi-head self-attention
        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.embedding_dim, dropout=self.dropout
        )(inputs, inputs)
        attention_output = layers.Dropout(self.dropout)(attention_output)
        out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed forward network
        ffn_output = layers.Dense(self.ff_dim, activation='relu')(out1)
        ffn_output = layers.Dense(self.embedding_dim)(ffn_output)
        ffn_output = layers.Dropout(self.dropout)(ffn_output)
        out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        
        return out2
    
    def _build_model(self):
        """Build Transformer model"""
        inputs = layers.Input(shape=(self.max_sequence_length,))
        
        # Embedding and positional encoding
        embedding = layers.Embedding(self.vocab_size, self.embedding_dim)(inputs)
        positions = tf.range(start=0, limit=self.max_sequence_length, delta=1)
        positional_encoding = layers.Embedding(self.max_sequence_length, self.embedding_dim)(positions)
        x = embedding + positional_encoding
        
        # Transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = self._transformer_block(x)
        
        # Global average pooling and classification
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(self.num_classes, 
                              activation='softmax' if self.num_classes > 2 else 'sigmoid')(x)
        
        return keras.Model(inputs, outputs)
    
    def compile_model(self, learning_rate=0.001, optimizer='adam'):
        """Compile Transformer model"""
        if optimizer == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
        else:
            loss = 'sparse_categorical_crossentropy'
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=['accuracy']
        )
    
    def get_model(self):
        """Return the Keras model"""
        return self.model

class ModelFactory:
    """Factory class for creating different TensorFlow/Keras models"""
    
    @staticmethod
    def create_model(model_type, config):
        """
        Create model based on type and configuration
        
        Args:
            model_type: Type of model ('mlp', 'cnn', 'lstm', 'transformer')
            config: Model configuration dictionary
        
        Returns:
            Compiled Keras model
        """
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        
        if model_type == 'mlp':
            classifier = MLPClassifier(
                input_size=model_config.get('input_size'),
                hidden_layers=model_config.get('hidden_layers', [128, 64]),
                output_size=model_config.get('output_size'),
                dropout_rate=model_config.get('dropout_rate', 0.2),
                activation=model_config.get('activation', 'relu'),
                batch_norm=model_config.get('batch_norm', False)
            )
        
        elif model_type == 'cnn':
            classifier = CNNClassifier(
                input_shape=model_config.get('input_shape'),
                num_classes=model_config.get('output_size'),
                architecture=model_config.get('architecture', 'simple')
            )
        
        elif model_type == 'lstm':
            classifier = LSTMClassifier(
                max_sequence_length=model_config.get('max_sequence_length', 100),
                vocab_size=model_config.get('vocab_size', 10000),
                embedding_dim=model_config.get('embedding_dim', 100),
                lstm_units=model_config.get('lstm_units', 128),
                num_classes=model_config.get('output_size'),
                dropout=model_config.get('dropout_rate', 0.2)
            )
        
        elif model_type == 'transformer':
            classifier = TransformerClassifier(
                max_sequence_length=model_config.get('max_sequence_length', 100),
                vocab_size=model_config.get('vocab_size', 10000),
                embedding_dim=model_config.get('embedding_dim', 128),
                num_heads=model_config.get('num_heads', 4),
                ff_dim=model_config.get('ff_dim', 128),
                num_transformer_blocks=model_config.get('num_transformer_blocks', 2),
                num_classes=model_config.get('output_size'),
                dropout=model_config.get('dropout_rate', 0.1)
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Compile model
        classifier.compile_model(
            learning_rate=training_config.get('learning_rate', 0.001),
            optimizer=training_config.get('optimizer', 'adam')
        )
        
        return classifier.get_model()

class ModelTrainer:
    """
    TensorFlow/Keras model trainer with advanced features
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Training configuration
        training_config = config.get('training', {})
        self.batch_size = training_config.get('batch_size', 32)
        self.epochs = training_config.get('epochs', 100)
        self.validation_split = training_config.get('validation_split', 0.2)
        
        # Setup callbacks
        self.callbacks = self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Setup training callbacks"""
        callback_list = []
        
        # Model checkpoint
        checkpoint = callbacks.ModelCheckpoint(
            filepath='./outputs/models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callback_list.append(checkpoint)
        
        # Early stopping
        early_stopping_config = self.config.get('training', {}).get('early_stopping', {})
        if early_stopping_config.get('enabled', True):
            early_stopping = callbacks.EarlyStopping(
                monitor=early_stopping_config.get('monitor', 'val_loss'),
                patience=early_stopping_config.get('patience', 10),
                restore_best_weights=early_stopping_config.get('restore_best_weights', True),
                verbose=1
            )
            callback_list.append(early_stopping)
        
        # Reduce learning rate on plateau
        lr_scheduler_config = self.config.get('training', {}).get('lr_scheduler', {})
        if lr_scheduler_config.get('type') == 'plateau':
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=lr_scheduler_config.get('factor', 0.5),
                patience=lr_scheduler_config.get('patience', 5),
                min_lr=lr_scheduler_config.get('min_lr', 1e-7),
                verbose=1
            )
            callback_list.append(reduce_lr)
        
        # TensorBoard logging
        if self.config.get('logging', {}).get('tensorboard', False):
            tensorboard = callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch'
            )
            callback_list.append(tensorboard)
        
        # Custom metrics logging
        csv_logger = callbacks.CSVLogger('./outputs/training_log.csv', append=True)
        callback_list.append(csv_logger)
        
        return callback_list
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        
        Returns:
            Training history
        """
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None
        else:
            validation_data = None
            validation_split = self.validation_split
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=self.callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        predictions = self.model.predict(X_test)
        
        if predictions.shape[1] == 1:
            # Binary classification
            pred_classes = (predictions > 0.5).astype(int).flatten()
        else:
            # Multi-class classification
            pred_classes = np.argmax(predictions, axis=1)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'predictions': pred_classes,
            'probabilities': predictions
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self.model

# Utility functions
def create_model_from_config(config_path):
    """Create model from configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_type = config.get('model', {}).get('type', 'mlp')
    return ModelFactory.create_model(model_type, config)

def plot_training_history(history):
    """Plot training history"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation accuracy values
    axes[0].plot(history.history['accuracy'])
    axes[0].plot(history.history['val_accuracy'])
    axes[0].set_title('Model accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'])
    axes[1].set_title('Model loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('./outputs/figures/training_history.png')
    plt.show()

if __name__ == "__main__":
    # Example usage
    config = {
        'model': {
            'type': 'mlp',
            'input_size': 10,
            'hidden_layers': [128, 64, 32],
            'output_size': 2,
            'dropout_rate': 0.2,
            'activation': 'relu',
            'batch_norm': True
        },
        'training': {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'optimizer': 'adam',
            'early_stopping': {
                'enabled': True,
                'patience': 10
            }
        }
    }
    
    # Create model
    model = ModelFactory.create_model('mlp', config)
    print(f"Model created with {model.count_params()} parameters")
    
    # Create trainer
    trainer = ModelTrainer(model, config)
    print("Trainer initialized successfully")