# Phonex-C1: Pure C Transformer Implementation

A minimal, hardened GPT-style transformer implemented entirely in C with no external dependencies beyond the standard library. This project demonstrates a complete training and inference pipeline for character-level language modeling.

## Features

- **Pure C Implementation**: No frameworks, no external dependencies (only `math.h` and standard library)
- **Full Transformer Architecture**: Multi-head self-attention, feed-forward networks, layer normalization, and RoPE positional embeddings
- **Training & Inference**: Complete forward/backward passes with AdamW optimization
- **Numerically Stable**: LogSumExp trick for softmax, proper gradient normalization, bound checking
- **Memory Efficient**: Static allocations with reusable cache structures

## Architecture

### Model Specifications

```
Layers:          4
Model Dimension: 64
Attention Heads: 4
FFN Dimension:   256 (4× model dim)
Sequence Length: 64
Vocabulary:      128 (ASCII)
Parameters:      ~150K
```

### Components

- **Token Embeddings**: Direct lookup table for ASCII characters
- **RoPE**: Rotary Position Embeddings (precomputed for efficiency)
- **Multi-Head Attention**: Scaled dot-product with causal masking
- **Feed-Forward Network**: Two-layer MLP with GELU activation
- **Layer Normalization**: Applied pre-attention and pre-FFN
- **AdamW Optimizer**: With bias correction and weight decay

## Installation

### Prerequisites

- GCC or any C compiler with C99 support
- Standard math library (`-lm`)

### Compilation

```bash
gcc -O3 c1p_harden.c -lm -o c1p
```

For debugging:
```bash
gcc -g -Wall c1p_harden.c -lm -o c1p
```

## Usage

### Training

Train the model on the built-in text sequence:

```bash
./c1p train
```

The model will:
- Train for 5,000 steps on character prediction
- Save checkpoints to `model.bin`
- Display loss every 100 steps

### Inference

Generate text starting from a prompt:

```bash
./c1p infer
```

Default prompt: "Hello"

The model will:
- Load weights from `model.bin`
- Generate up to 64 characters
- Use temperature sampling (0.8) with top-p nucleus sampling (0.9)

## Code Structure

### Core Data Structures

```c
Tensor          // Holds data, gradients, and Adam moments
Block           // Single transformer layer (attention + FFN)
GPT             // Complete model with embeddings and layers
GPTCache        // Activation cache for forward/backward passes
LayerCache      // Per-layer intermediate activations
```

### Key Functions

**Forward Pass:**
- `forward()` - Main forward propagation
- `attention_forward()` - Multi-head self-attention
- `layer_norm_forward()` - Layer normalization
- `apply_rope_forward()` - Rotary embeddings

**Backward Pass:**
- `backward()` - Main backpropagation
- `attention_backward()` - Attention gradients
- `backward_layer_norm()` - Layer norm gradients
- `backward_linear()` - Linear layer gradients

**Optimization:**
- `adamw_step()` - AdamW parameter updates
- `optimizer_step()` - Full model update

**Sampling:**
- `sample_top_p()` - Nucleus sampling for generation

## Hyperparameters

Edit these constants in the source code:

```c
#define LR 0.001f           // Learning rate
#define WEIGHT_DECAY 0.01f  // L2 regularization
#define BETA1 0.9f          // Adam beta1
#define BETA2 0.99f         // Adam beta2
#define TRAIN_STEPS 5000    // Training iterations
#define TEMP 0.8f           // Sampling temperature
#define TOP_P 0.9f          // Nucleus sampling threshold
```

## Memory Requirements

Approximate memory usage:

- **Model Parameters**: ~600 KB (150K params × 4 bytes)
- **Gradients**: ~600 KB
- **Adam Moments**: ~1.2 MB (m and v)
- **Activation Cache**: ~2 MB
- **Total**: ~4.5 MB

## Training Data

The default training corpus is a simple string:
```c
char* text = "Hello World! Phonex-C1 Refactored. ";
```

To train on custom data:
1. Replace the `text` variable in `main()`
2. Ensure text length > SEQ_LEN (64 characters)
3. Keep characters within ASCII range (0-127)

For larger datasets, modify the code to:
- Load text from file
- Implement proper batching
- Add data preprocessing

## Technical Details

### Numerical Stability

- **LogSumExp**: Prevents overflow in softmax computation
- **Gradient Clipping**: Implicit through normalization
- **Xavier Initialization**: Proper weight scaling for deep networks
- **Epsilon Guards**: Prevent division by zero in layer norm

### Gradient Flow

The implementation ensures proper gradient routing through:
- Residual connections (attention + FFN)
- Layer normalization backprop with correct statistics
- RoPE rotation in both directions (forward/backward)
- Attention mask handling in gradient computation

### Optimizations

- **Memory Reuse**: Single cache structure for all forward passes
- **Precomputed RoPE**: Sin/cos tables calculated once at initialization
- **In-place Operations**: Where possible to reduce allocations
- **Static Shapes**: All tensor sizes known at compile time

## Limitations

- **Single Character Vocabulary**: Limited to ASCII (0-127)
- **No Batching**: Processes one sequence at a time
- **No KV Cache**: Full recomputation for each generation step
- **Fixed Context**: Maximum 64 tokens
- **CPU Only**: No GPU acceleration

## Extending the Model

### Increase Model Capacity

```c
#define D_MODEL 128        // Wider model
#define N_HEADS 8          // More attention heads
#define MAX_LAYERS 8       // Deeper network
```

### Add Byte-Pair Encoding

Replace character-level tokenization with BPE:
1. Implement tokenizer in preprocessing
2. Increase `VOCAB_SIZE` accordingly
3. Add token-to-string mapping for generation

### Implement KV Cache

For faster inference:
1. Store key/value tensors from previous tokens
2. Only compute attention for new tokens
3. Concatenate with cached keys/values

## Performance

On modern hardware (approximate):

- **Training**: ~100-500 steps/second (CPU-dependent)
- **Inference**: ~10-50 tokens/second
- **Convergence**: Loss < 1.0 after 2-3K steps on simple text

## Troubleshooting

**Loss exploding/NaN:**
- Reduce learning rate
- Check for gradient accumulation bugs
- Verify input data is in valid range

**Poor generation quality:**
- Train for more steps
- Increase model size
- Use more diverse training data
- Adjust temperature/top-p values

**Compilation errors:**
- Ensure C99 or later standard
- Link math library with `-lm`
- Check all dependencies are available

