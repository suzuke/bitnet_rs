
# BitNet
This is an unofficial implementation of: [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453) and [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764).

BitLinear and BitLinear1_58 are candle modules for training and inferencing binarized linear layers, which is a key component of the [BitNet](https://arxiv.org/abs/2310.11453) and [BitNet1.58](https://arxiv.org/abs/2402.17764) architecture. It can effectively reduce the memory footprint and energy consumption of large language models while maintaining competitive performance compared to full-precision models.

## Features

- Quantization of weights and activations to a specified bit-width
- Support for bias addition and optional use of bias
- Option to apply quantization before or after the non-linear activation
- Configurable parameters for bit-width, epsilon value, and quantization method

## Usage

1. Import the necessary modules:

```rust
use candle_core::{Result, Tensor};
use candle_nn::{VarBuilder, init};
use bitnet_rs::bitlinear::{bitlinear, BitLinearConfig};
```

2. Create a `BitLinearConfig` with your desired settings:

```rust
let config = BitLinearConfig {
    bit_width: 8,
    eps: 1e-5,
    bias: true,
    use_before_nonlinear: false,
};
```

3. Create a `BitLinear` layer using the `bitlinear` function:

```rust
let bitlinear = bitlinear(input_dim, output_dim, config, var_builder)?;
```

4. Forward pass through the `BitLinear` layer:

```rust
let output = bitlinear.forward(&input_tensor)?;
```

## Configuration

The `BitLinearConfig` struct allows you to customize the behavior of the `BitLinear` layer:

- `bit_width`: The number of bits to use for quantization (default: 8).
- `eps`: A small value used to clamp the quantized values (default: 1e-5).
- `bias`: Whether to include a bias term (default: true).
- `use_before_nonlinear`: Whether to apply quantization before or after the non-linear activation (default: false).

## Examples

You can find examples of using `BitLinear` and `BitLinear1_58` in the [examples](./examples/).

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Citation

If you use BitLinear in your research, please cite the following paper:

BitNet: Scaling 1-bit Transformers for Large Language Models
```
@article{wang2023bitnet,
  title={BitNet: Scaling 1-bit Transformers for Large Language Models},
  author={Wang, Hongyu and Ma, Shuming and Dong, Li and Huang, Shaohan and Wang, Huaijie and Ma, Lingxiao and Yang, Fan and Wang, Ruiping and Wu, Yi and Wei, Furu},
  journal={arXiv preprint arXiv:2310.11453},
  year={2023}
}
```

The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits
```
@article{ma2024era,
title={The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits},
author={Ma, Shuming and Wang, Hongyu and Ma, Lingxiao and Wang, Lei and Wang, Wenhui and Huang, Shaohan and Dong, Li and Wang, Ruiping and Xue, Jilong and Wei, Furu},
journal={arXiv preprint arXiv:2402.17764},
year={2024}
}
```
