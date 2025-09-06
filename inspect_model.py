"""
Script to inspect and analyze the .h5 model file
"""
import os
import h5py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import tensorflow as tf

def check_file_stats(model_path):
    """Check basic file statistics"""
    print("=" * 50)
    print("FILE STATISTICS")
    print("=" * 50)
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"File exists: ✅")
        print(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
        print(f"File path: {os.path.abspath(model_path)}")
        
        # Check file permissions
        print(f"Readable: {'✅' if os.access(model_path, os.R_OK) else '❌'}")
        print(f"Writable: {'✅' if os.access(model_path, os.W_OK) else '❌'}")
    else:
        print(f"File does not exist: ❌")
        return False
    return True

def inspect_h5_structure(model_path):
    """Inspect the HDF5 file structure"""
    print("\n" + "=" * 50)
    print("HDF5 FILE STRUCTURE")
    print("=" * 50)
    
    try:
        with h5py.File(model_path, 'r') as f:
            print(f"HDF5 file keys: {list(f.keys())}")
            
            def print_structure(name, obj):
                print(f"  {name}: {type(obj).__name__}")
                if hasattr(obj, 'shape'):
                    print(f"    Shape: {obj.shape}")
                if hasattr(obj, 'dtype'):
                    print(f"    Dtype: {obj.dtype}")
            
            print("\nDetailed structure:")
            f.visititems(print_structure)
            
    except Exception as e:
        print(f"Error reading HDF5 structure: {e}")

def inspect_keras_model(model_path):
    """Load and inspect the Keras model"""
    print("\n" + "=" * 50)
    print("KERAS MODEL ANALYSIS")
    print("=" * 50)
    
    try:
        # Load model
        model = load_model(model_path, compile=False)
        print("Model loaded successfully! ✅")
        
        # Basic model info
        print(f"\nModel type: {type(model).__name__}")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        
        # Model summary
        print("\n" + "-" * 30)
        print("MODEL SUMMARY")
        print("-" * 30)
        model.summary()
        
        # Layer information
        print(f"\nTotal layers: {len(model.layers)}")
        print("\nLayer details:")
        for i, layer in enumerate(model.layers):
            print(f"  {i+1}. {layer.name} ({type(layer).__name__})")
            if hasattr(layer, 'output_shape'):
                print(f"     Output shape: {layer.output_shape}")
            if hasattr(layer, 'count_params'):
                print(f"     Parameters: {layer.count_params():,}")
        
        # Total parameters
        total_params = model.count_params()
        trainable_params = sum([layer.count_params() for layer in model.layers if layer.trainable])
        non_trainable_params = total_params - trainable_params
        
        print(f"\nParameter summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable parameters: {non_trainable_params:,}")
        
        # Model configuration
        print(f"\nModel configuration:")
        config = model.get_config()
        print(f"  Model name: {config.get('name', 'N/A')}")
        
        # Check if model has weights
        try:
            weights = model.get_weights()
            print(f"  Number of weight arrays: {len(weights)}")
            total_weight_elements = sum([w.size for w in weights])
            print(f"  Total weight elements: {total_weight_elements:,}")
        except Exception as e:
            print(f"  Error getting weights: {e}")
            
        return model
        
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return None

def analyze_model_architecture(model):
    """Analyze model architecture in detail"""
    if model is None:
        return
        
    print("\n" + "=" * 50)
    print("DETAILED ARCHITECTURE ANALYSIS")
    print("=" * 50)
    
    # Input/Output analysis
    print("Input/Output Analysis:")
    print(f"  Input dtype: {model.input.dtype}")
    print(f"  Output dtype: {model.output.dtype}")
    
    # Check for common layer types
    layer_types = {}
    for layer in model.layers:
        layer_type = type(layer).__name__
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
    
    print(f"\nLayer type distribution:")
    for layer_type, count in sorted(layer_types.items()):
        print(f"  {layer_type}: {count}")
    
    # Check activation functions
    activations = set()
    for layer in model.layers:
        if hasattr(layer, 'activation') and layer.activation is not None:
            activations.add(layer.activation.__name__)
    
    if activations:
        print(f"\nActivation functions used: {', '.join(activations)}")
    
    # Model complexity metrics
    print(f"\nModel complexity:")
    print(f"  Depth (number of layers): {len(model.layers)}")
    
    # Try to estimate model type
    if any('conv' in layer.name.lower() for layer in model.layers):
        print(f"  Model type: Likely CNN (Convolutional Neural Network)")
    elif any('lstm' in layer.name.lower() or 'gru' in layer.name.lower() for layer in model.layers):
        print(f"  Model type: Likely RNN (Recurrent Neural Network)")
    else:
        print(f"  Model type: Likely Dense/Feedforward Network")

def test_model_prediction(model, model_path):
    """Test model with dummy input"""
    if model is None:
        return
        
    print("\n" + "=" * 50)
    print("MODEL PREDICTION TEST")
    print("=" * 50)
    
    try:
        # Create dummy input
        input_shape = model.input_shape[1:]  # Remove batch dimension
        dummy_input = np.random.random((1,) + input_shape).astype(np.float32)
        
        print(f"Testing with dummy input shape: {dummy_input.shape}")
        
        # Make prediction
        prediction = model.predict(dummy_input, verbose=0)
        
        print(f"Prediction output shape: {prediction.shape}")
        print(f"Prediction values: {prediction[0]}")
        print(f"Prediction sum: {np.sum(prediction[0]):.6f}")
        print(f"Max prediction: {np.max(prediction[0]):.6f}")
        print(f"Min prediction: {np.min(prediction[0]):.6f}")
        
        # Check if it looks like probabilities
        if np.allclose(np.sum(prediction[0]), 1.0, atol=1e-3):
            print("✅ Output appears to be normalized probabilities (softmax)")
        else:
            print("ℹ️  Output does not sum to 1 (may need softmax activation)")
            
        # Predicted class
        predicted_class = np.argmax(prediction[0])
        print(f"Predicted class index: {predicted_class}")
        
    except Exception as e:
        print(f"Error during prediction test: {e}")

def main():
    model_path = "coconut_leaf_disease_model.h5"
    
    print("🔍 COMPREHENSIVE MODEL INSPECTION")
    print("=" * 60)
    
    # Check file stats
    if not check_file_stats(model_path):
        return
    
    # Inspect HDF5 structure
    inspect_h5_structure(model_path)
    
    # Load and analyze Keras model
    model = inspect_keras_model(model_path)
    
    # Detailed architecture analysis
    analyze_model_architecture(model)
    
    # Test prediction
    test_model_prediction(model, model_path)
    
    print("\n" + "=" * 60)
    print("✅ INSPECTION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
