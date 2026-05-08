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

### The Input: 1536 Dimensions of Perspective
To evaluate a chess position, the network first needs to "see" the board. We represent the board using a 1536-feature input array. This is essentially two 768-dimensional perspective arrays (one from our perspective, one from the enemy's perspective) concatenated together right at the input level. 

There are 64 squares on a chessboard, and 12 possible piece types. `12 * 64 = 768`. We use a one-hot encoding system to represent the specific (piece, square) combination.

* **Our Pieces:** Pawn (0), Knight (1), Bishop (2), Rook (3), Queen (4), King (5)
* **Enemy Pieces:** Offset by 6 (e.g., Enemy Pawn is 6)

Starting from the upper-left square, the exact index for any piece on the board is calculated using this formula:
> **Location = 64 * (piece_encoding) + 8 * (row) + column**

### Projection & The Flat 2D Flattening
The 1536-wide input is split into its two respective halves. We project each 768-wide perspective onto a massive 2048-dimensional space. While this sounds computationally heavy, it allows the network to capture complex, non-linear piece interactions early on.

Originally, we planned to reshape this data into a 3D tensor and use channel-wise convolutions. However, edge hardware compilers are notoriously strict. Complex reshapes often trigger unsupported operation errors (like the dreaded "Op 152" on the Coral TPU), causing the compiler to kick operations back to the host CPU.

To ensure 100% TPU compatibility, we keep the architecture entirely flat. We pass that 2048-dimensional output directly into another massive Dense (Linear) layer to condense the data down into a refined **256-wide vector**. Keeping the math strictly 2D guarantees the silicon will accept it.

### Perspective Merging
Chess is a game of perfect information, and understanding who is winning requires looking at the board from both sides. We take the 256-wide vector evaluated from our perspective and concatenate it with the 256-wide vector evaluated from the opponent's perspective. We now have a **512-wide vector** that contains a complete, balanced understanding of the board's tension.

### Hidden Layers, Skip Connections, and the Self-Square Trick
Now we narrow this 512-wide vector down to a single evaluation score using dense layers and a ResNet-style skip connection:

1.  **The Skip Connection:** We pass the 512-wide perspective vector through a direct bypass layer straight to the final output, preserving the raw, high-dimensional evaluation.
2.  **First Reduction:** Simultaneously, the 512-wide vector passes through a dense layer, reducing it to a **32-wide vector**.
3.  **The "Self-Square" Activation Trick:** We take this 32-wide vector and concatenate it with its own mathematically squared values (x²), scaled specifically by a factor of **127/128**. This creates a custom, quantization-friendly non-linear activation.
4.  **Second Reduction:** This newly concatenated vector passes through another dense layer, bringing it back down to a refined **32-wide vector**.
5.  **Final Output:** A final layer processes this 32-wide vector, and we add that result directly to the evaluation from our skip connection. 

Throughout this pipeline, every intermediate layer uses a **Clipped ReLU** activation function that strictly bounds values between 0 and 1 to prevent overflow when deploying on integer-only hardware.

<img width="541" height="821" alt="Architecture Flowchart" src="https://github.com/user-attachments/assets/f2e2ae10-22b3-43d4-8aad-416e4376863e" />

---

## 2. Dataset Generation and Edge-Aware Training

A brilliant, empty structure still doesn't know how to play chess. We need to teach it using a massive amount of high-quality data.

### The Lc0 Dataset & Knowledge Distillation
We accelerate the process by utilizing the open **Leela Chess Zero (Lc0) database**, giving our network a robust foundation of pre-played, high-level chess positions. 

We use Stockfish as our absolute ground truth. During PyTorch training, our custom CNN evaluates a board state and compares its answer to Stockfish's superhuman evaluation. Using **Mean Squared Error (MSE) loss** and backpropagation, the lightweight CNN slowly learns to mimic Stockfish's judgment.

### Training with an 8-Bit Destiny
We train our model knowing its eventual destiny is the Coral USB Accelerator's Int8 Arithmetic Logic Units. 

By strictly using **Clipped ReLU** (bounding outputs between 0 and 1) during every single epoch, we force the network to learn within a highly compressed mathematical space. The network physically cannot develop massive, outlier weights. When we later compress these numbers into 8-bit integers, we suffer almost zero quantization loss.

### The Export Pipeline: Manual Graph Reconstruction via Google Colab
To transition from software to hardware execution, the model goes through a strict conversion gauntlet. Standard converters like ONNX often fail or trigger unsupported operation errors when dealing with complex edge quantization rules. To bypass this entirely, we moved our compilation pipeline to **Google Colab** and utilized a hardcore hardware-software approach: Manual Weight Injection.

1. **Rebuilding in Keras:** Instead of relying on an automated translator, we manually reconstructed the exact same flattened 2D architecture natively in TensorFlow/Keras within a Colab notebook.
2. **Manual Weight Injection:** We loaded our trained PyTorch dictionary (`.pt`), extracted the raw numerical weights and biases, and forcefully injected them into the matching Keras layers. 
3. **Post-Training Quantization (TFLite):** We pushed the TensorFlow model through the TFLite converter. The converter squashes every Float32 weight into a static Int8 value using a small representative dataset to calibrate the bounds, generating a temporary `.tflite` file.
4. **Cloud TPU Compilation:** Finally, we ran this temporary `.tflite` file through the Edge TPU Compiler directly inside the Colab environment. Because our Keras graph perfectly matched hardware-supported operations, the compiler mapped 100% of our distilled chess intelligence directly onto the physical silicon.

We then simply download the resulting `_edgetpu.tflite` file from Colab and drop it locally into our Python backend for `app.py` to use.

---

## 3. Hooking the Edge TPU into the Pygame Frontend

Now, it is time for the grand finale: wiring the physical Google Coral USB Accelerator directly into our UI.

<img width="1089" height="1502" alt="Google Coral Edge TPU" src="https://github.com/user-attachments/assets/2bbc4afc-5058-4253-8efa-01e883603b5f" />

### Initializing the Hardware Delegate
To communicate with the Coral Accelerator, we use the lightweight `tflite_runtime` library. By loading the compiled model with the **Edge TPU Delegate**, we explicitly tell the Python interpreter to hand off the computations to the physical USB device using the `libedgetpu` library.

### The Search Loop: Picking the Best Move
When the human player makes a valid move in the Pygame window, it becomes the AI's turn. We handle the response through a hardware-accelerated search loop:

1. **Generate Legal Moves:** The script asks `python-chess` for every possible legal move the AI can make.
2. **Simulate & Translate:** For each move, we temporarily push it to the board, and translate the new board state into our 1536-dimensional tensor array (properly encoding both the active and flipped perspectives).
3. **Inference:** We send the tensor over USB. The Coral TPU instantly crunches the quantized weights and returns a single evaluation score.
4. **Select:** The engine keeps track of the move that yielded the highest evaluation score and pushes that optimal move to the real board.

### The Final User Experience

<img width="575" height="598" alt="Pygame Interface" src="https://github.com/user-attachments/assets/05979349-afb2-4bbd-b75d-c0ce337dace7" />

Once this logic is slotted into the Pygame main loop, the user experience is seamless. You click your White Pawn, drag it forward two squares, and release it. Instantly, Python updates the board state, generates the AI's possible responses, fires the tensors over the USB cable to the Coral Accelerator, retrieves the evaluations, picks the optimal move, and updates the UI. 

Driven entirely by a custom neural network running on low-power edge hardware—no cloud APIs, no massive desktop GPUs—just pure, localized edge intelligence.

---

## Setup and Installation

Follow these steps to set up the hardware, install the dependencies, and run the edge-accelerated chess engine on your local machine.

### Step 1: Hardware Prerequisites
1. Obtain a **Google Coral USB Accelerator**.
2. Install the official Edge TPU Runtime for your specific operating system (Windows, Linux, or macOS). 
3. Plug the Coral USB Accelerator into an available **USB 3.0** port on your computer. 
### Step 2: Clone the Repository
Open your terminal or command prompt and clone this repository to your local machine:
```bash
git clone [https://github.com/YourUsername/Project-ICT740.git](https://github.com/YourUsername/Project-ICT740.git)
cd Project-ICT740
```
### Step 3: Install Software Dependencies
```bash
pip install pygame chess numpy torch tflite_runtime
```
### Step 4: Run the app
```bash
python app.py
```
