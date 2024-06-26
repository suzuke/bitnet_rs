# MNIST Training Example

This repository contains an example implementation of training models on the MNIST dataset using candle, a deep learning library.

## Credits
The core of this code has been adapted from [candle](https://github.com/huggingface/candle), specifically the MNIST training example available [here](https://github.com/huggingface/candle/blob/455c42aa729d8019fcb496106478e75dd3246c08/candle-examples/examples/mnist-training/main.rs).

## Models and Performance
- **Linear Model**: Achieves an accuracy of 91.5%
- **MLP (Multi-Layer Perceptron) Model**: Achieves an accuracy of 88.4%
- **CNN (Convolutional Neural Network) Model**: 

- **Bit-Linear Model**: Achieves an accuracy of 85.8%
- **Bit-MLP Model**: Achieves an accuracy of 88.2%
- **Bit-CNN Model**: 

- **Bit-Linear 1.58-bit Model** Archieves an accuracy of 85.1%
- **Bit-MLP 1.58-bit Model** Archieves an accuracy of 93.9%

## Dependencies
- `clap` for command-line argument parsing
- `rand` for random number generation
- `candle_core` and `candle_nn` for deep learning operations
- `bitnet_rs` for bit operations in neural networks

## Usage
To train different models on the MNIST dataset, you can run the main function. You can specify various options such as learning rate, number of epochs, and whether to save or load trained weights.

```bash
cargo run -- [MODEL] [OPTIONS]
```

### MODEL
- `linear`: Linear Model
- `mlp`: MLP(Multi-Layer Perceptron) Model
- `cnn`: CNN (Convolutional Neural Network) Model
- `bit-linear`: Bit-Linear Model
- `bit-mlp`: Bit-MLP Model
- `bit-cnn`: Bit-CNN Model
- `bit-linear1_58`: 1.58-Bit-Linear Model
- `bit-mlp1_58`: 1.58-Bit-MLP Model


### Options
- `--learning_rate`: Learning rate for training. Default values are provided for each model.
- `--epochs`: Number of epochs for training. Default is 200.
- `--save`: File to save trained weights.
- `--load`: File to load pre-trained weights.
- `--local_mnist`: Directory to load the MNIST dataset from in ubyte format.

## Example
```bash
cargo run -r -- bit-linear --learning-rate 0.1 --epochs 100 --save bitlinear_model_weights.bin
```

This command will train a Linear model with a learning rate of 0.1 for 100 epochs and save the trained weights in the file `linear_model_weights.bin`.