import numpy as np
import tensorflow as tf

# 1. Path to your model and the Edge TPU driver
model_path = 'ready_for_compiler_edgetpu.tflite'
# On Windows, this DLL was installed by the install.bat you ran earlier
delegate_path = 'edgetpu.dll' 

try:
    # 2. Load the interpreter with the TPU delegate
    print("Connecting to Edge TPU...")
    interpreter = tf.lite.Interpreter(
        model_path=model_path,
        experimental_delegates=[tf.lite.load_delegate(delegate_path)]
    )
    interpreter.allocate_tensors()
    print("SUCCESS: Model loaded onto the Coral TPU!")

    # 3. Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 4. Prepare a dummy input (1536 integers)
    # Ensure it is int8 since we quantized the model
    test_input = np.zeros((1, 1536), dtype=np.int8)

    # 5. Run a test inference
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    # 6. Get the result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(f"TPU Output: {output_data}")
    print("Inference successful!")

except Exception as e:
    print(f"Initialization failed: {e}")
    print("\nTroubleshooting:")
    print("- Ensure the Coral USB is plugged into a USB 3.0 port.")
    print("- Make sure 'edgetpu.dll' is in your System32 folder or the current directory.")