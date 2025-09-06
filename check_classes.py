"""
Quick script to check model output classes
"""
import numpy as np
from tensorflow.keras.models import load_model

def check_model_classes():
    model_path = "coconut_leaf_disease_model.h5"
    
    try:
        # Load model
        print("Loading model...")
        model = load_model(model_path, compile=False)
        
        # Get output shape
        output_shape = model.output_shape
        num_classes = output_shape[-1]
        
        print(f"Model output shape: {output_shape}")
        print(f"Number of classes: {num_classes}")
        
        # Test with dummy input to see output
        input_shape = model.input_shape[1:]  # Remove batch dimension
        dummy_input = np.random.random((1,) + input_shape).astype(np.float32)
        
        prediction = model.predict(dummy_input, verbose=0)
        print(f"Sample prediction output: {prediction[0]}")
        print(f"Predicted class index: {np.argmax(prediction[0])}")
        
        # Check if probabilities sum to 1
        prob_sum = np.sum(prediction[0])
        print(f"Prediction sum: {prob_sum:.6f}")
        
        if np.allclose(prob_sum, 1.0, atol=1e-3):
            print("✅ Output is normalized (softmax)")
        else:
            print("ℹ️ Output is not normalized")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_model_classes()
