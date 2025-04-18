from tensorflow.keras.models import load_model

# Load your model
model = load_model("backend/model/lstm_mobilenet_skeleton.h5")

# Print a summary of the model
model.summary()

# Access input and output shapes
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

# Print model configuration
#print("\nModel Config:")
#print(model.get_config())

# Optionally: print layer names and shapes
#print("\nLayer details:")
#for i, layer in enumerate(model.layers):
#    print(f"{i+1}. {layer.name} - {layer.output_shape}")
