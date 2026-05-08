# Project-ICT740
# Edge-Accelerated Chess Engine Using Google Coral TPU

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Model_Training-EE4C2C.svg)
![TensorFlow Lite](https://img.shields.io/badge/TFLite-Edge_TPU_Compiled-FF6F00.svg)
![Pygame](https://img.shields.io/badge/Pygame-Frontend-green.svg)

Modern chess engines like Stockfish achieve superhuman performance by combining deep search algorithms with highly optimized neural network evaluations (NNUE). However, these engines are typically designed to run on powerful desktop CPUs. Deploying complex AI models locally on low-power IoT devices or edge accelerators usually results in severe bottlenecks or limited playing strength. 

To solve this, we are exploring a hardware-software co-design approach. By building a custom, lightweight Convolutional Neural Network (CNN) tailored for hardware accelerators, we can offload the heavy lifting of board evaluation to a physical edge AI ASIC.

---

## 1. Demystifying the Neural Network Architecture

Before we get into the communication protocols or the user interface, here is a breakdown of the specific, highly quantization-friendly network architecture we use to evaluate chess positions on the fly.

### The Input: 768 Dimensions of Perspective
To evaluate a chess position, the network first needs to "see" the board. We represent the board from a specific player's perspective using a 768-feature input array. There are 64 squares on a chessboard, and 12 possible piece types. `12 * 64 = 768`. We use a one-hot encoding system to represent the specific (piece, square) combination.

* **Our Pieces:** Pawn (0), Knight (1), Bishop (2), Rook (3), Queen (4), King (5)
* **Enemy Pieces:** Offset by 6 (e.g., Enemy Pawn is 6)

Starting from the upper-left square, the exact index for any piece on the board is calculated using this formula:
> **Location = 64 * (piece_encoding) + 8 * (row) + column**

### Projection & Perspective Merging
Once we have our 768 input features, we project them onto a massive 2048-dimensional space, reshaping it into an `(8, 256)` tensor. We then apply a convolution of size 1 across this tensor to combine the results channel-wise into a dense **256-wide vector**.

We take this vector (evaluated from our perspective) and concatenate it with a similarly processed 256-wide vector evaluated from the opponent's perspective. We now have a **512-wide vector** that contains a complete, balanced understanding of the board's tension.

### Hidden Layers, Skip Connections, and the Self-Square Trick
Now we narrow this 512-wide vector down to a single evaluation score using dense layers and a ResNet-style skip connection:

1.  **The Skip Connection:** We pass the 512-wide perspective vector through a direct bypass layer straight to the final output, preserving the raw, high-dimensional evaluation.
2.  **First Reduction:** Simultaneously, the 512-wide vector passes through a dense layer, reducing it to a **32-wide vector**.
3.  **The "Self-Square" Activation Trick:** We take this 32-wide vector and concatenate it with its own mathematically squared values (x²), scaled specifically by a factor of **127/128**. This creates a custom, quantization-friendly non-linear activation.
4.  **Second Reduction:** This newly concatenated vector passes through another dense layer, bringing it back down to a refined **32-wide vector**.
5.  **Final Output:** A final layer processes this 32-wide vector, and we add that result directly to the evaluation from our skip connection. 

Throughout this pipeline, every intermediate layer uses a **Clipped ReLU** activation function that strictly bounds values between 0 and 1 to prevent overflow when deploying on integer-only hardware.

<img width="541" height="821" alt="Diagram drawio" src="https://github.com/user-attachments/assets/f2e2ae10-22b3-43d4-8aad-416e4376863e" />
---

## 2. Dataset Generation and Edge-Aware Training

A brilliant, empty structure still doesn't know how to play chess. We need to teach it using a massive amount of high-quality data.

### The Lc0 Dataset & Knowledge Distillation
We accelerate the process by utilizing the open **Leela Chess Zero (Lc0) database**, giving our network a robust foundation of pre-played, high-level chess positions. 

We use Stockfish as our absolute ground truth. During PyTorch training, our custom CNN evaluates a board state and compares its answer to Stockfish's superhuman evaluation. Using **Mean Squared Error (MSE) loss** and backpropagation, the lightweight CNN slowly learns to mimic Stockfish's judgment.

### Training with an 8-Bit Destiny
We train our model knowing its eventual destiny is the Coral USB Accelerator's Int8 Arithmetic Logic Units. 

By strictly using **Clipped ReLU** (bounding outputs between 0 and 1) during every single epoch, we force the network to learn within a highly compressed mathematical space. The network physically cannot develop massive, outlier weights. When we later compress these numbers into 8-bit integers, we suffer almost zero quantization loss.

### The Export Pipeline
To transition from software to hardware execution, the model goes through a strict conversion gauntlet:

1. **PyTorch to ONNX:** We export our model to the Open Neural Network Exchange format, isolating the pure mathematical graph.
2. **ONNX to TensorFlow:** We use `onnx-tf` to translate the graph into a native TensorFlow SavedModel.
3. **Post-Training Quantization (TFLite):** The TFLite converter squashes every Float32 weight into a static Int8 value using a small representative dataset of Lc0 positions to calibrate the bounds.
4. **The Edge TPU Compiler:** We compile the `.tflite` file. Because we restricted our architecture to standard convolutions and dense layers, the compiler accepts 100% of the operations.

The result is a tiny `_edgetpu.tflite` file, ready for edge deployment.

---

## 3. Hooking the Edge TPU into the Pygame Frontend

Now, it is time for the grand finale: wiring the physical Google Coral USB Accelerator directly into our UI.

<img width="1089" height="1502" alt="1448" src="https://github.com/user-attachments/assets/2bbc4afc-5058-4253-8efa-01e883603b5f" />

### Initializing the Hardware Delegate
To communicate with the Coral Accelerator, we use the lightweight `tflite_runtime` library. By loading the compiled model with the **Edge TPU Delegate**, we explicitly tell the Python interpreter to hand off the computations to the physical USB device using the `libedgetpu` library.

### The Search Loop: Picking the Best Move
When the human player makes a valid move in the Pygame window, it becomes the AI's turn. We handle the response through a hardware-accelerated search loop:

1. **Generate Legal Moves:** The script asks `python-chess` for every possible legal move the AI can make.
2. **Simulate & Translate:** For each move, we temporarily push it to the board, and translate the new board state into our 768-dimensional tensor array.
3. **Inference:** We send the tensor over USB. The Coral TPU instantly crunches the quantized weights and returns a single evaluation score.
4. **Select:** The engine keeps track of the move that yielded the highest evaluation score and pushes that optimal move to the real board.

### The Final User Experience

<img width="575" height="598" alt="Chess png" src="https://github.com/user-attachments/assets/05979349-afb2-4bbd-b75d-c0ce337dace7" />

Once this logic is slotted into the Pygame main loop, the user experience is seamless. You click your White Pawn, drag it forward two squares, and release it. Instantly, Python updates the board state, generates the AI's possible responses, fires the tensors over the USB cable to the Coral Accelerator, retrieves the evaluations, picks the optimal move, and updates the UI. 

Driven entirely by a custom neural network running on low-power edge hardware—no cloud APIs, no massive desktop GPUs—just pure, localized edge intelligence.
