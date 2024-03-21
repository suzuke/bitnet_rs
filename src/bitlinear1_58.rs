use candle_core::{Result, Tensor};
use candle_nn::{init, Init, Linear, Module, VarBuilder};

use crate::utils::*;

#[derive(Debug, Clone, Copy)]
pub struct BitLinear1_58Config {
    pub bit_width: usize,
    pub eps: f64,
    pub bias: bool,
}

impl Default for BitLinear1_58Config {
    fn default() -> Self {
        Self {
            bit_width: 8,
            eps: 1e-5,
            bias: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BitLinear1_58 {
    weight: Tensor,
    bias: Option<Tensor>,
    q_b: f64,
    config: BitLinear1_58Config,
}

impl BitLinear1_58 {
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: BitLinear1_58Config) -> Self {
        let q_b = 2.0_f64.powi(config.bit_width as i32 - 1);
        Self {
            weight,
            bias,
            q_b,
            config,
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    fn binarize_weight(&self) -> Result<Tensor> {
        let gamma = gamma1_58(&self.weight)?;
        let binarized_weight = round_clip_ste(&self.weight.broadcast_div(&(&gamma + self.config.eps)?)?, -1.0, 1.0);
        binarized_weight
    }

    fn quantize(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let gamma = gamma(x)?;
        let quantized_x = (x.broadcast_div(&gamma)? * self.q_b)?
            .clamp(-self.q_b + self.config.eps, self.q_b - self.config.eps)?;
        Ok((quantized_x, gamma))
    }

    fn dequantize(&self, x: &Tensor, gamma: &Tensor) -> Result<Tensor> {
        let beta = beta(&self.weight)?;
        let dequantized_output = x.broadcast_mul(&beta)?.broadcast_mul(&gamma)? / self.q_b;
        dequantized_output
    }
}

impl Module for BitLinear1_58 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = sub_ln(x, self.config.eps)?;

        let binarized_weight = self.binarize_weight()?;

        let (quantized_x, gamma) = self.quantize(&x)?;

        let y = Linear::new(binarized_weight, self.bias.clone()).forward(&quantized_x)?;

        let y = self.dequantize(&y, &gamma)?;

        Ok(y)
    }
}

pub fn bitlinear1_58<C: Into<BitLinear1_58Config>>(
    in_dim: usize,
    out_dim: usize,
    config: C,
    vb: VarBuilder,
) -> Result<BitLinear1_58> {
    let config = config.into();
    let weight = vb.get_with_hints((out_dim, in_dim), "weight", init::DEFAULT_KAIMING_NORMAL)?;
    let bound = 1. / (in_dim as f64).sqrt();
    let init_bs = Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bias = config
        .bias
        .then(|| vb.get_with_hints(out_dim, "bias", init_bs).unwrap());
    Ok(BitLinear1_58::new(weight, bias, config))
}


#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    const DEVICE : &Device = &Device::Cpu;

    #[test]
    fn test_bitlinear1_58() {
        let batch_dim = 1;
        let in_dim = 4;
        let out_dim = 3;
        let weight = Tensor::ones((out_dim, in_dim), DType::F32, DEVICE).unwrap();
        let bias = Tensor::zeros(out_dim, DType::F32, DEVICE).unwrap();
        let config = BitLinear1_58Config::default();
        let bitlinear = BitLinear1_58::new(weight.clone(), Some(bias), config);

        let input = Tensor::ones((batch_dim, in_dim), DType::F32, DEVICE).unwrap();
        let output = bitlinear.forward(&input).unwrap();

        assert_eq!(bitlinear.weight().to_vec2::<f32>().unwrap(), weight.to_vec2::<f32>().unwrap());
        assert_eq!(output.dims().to_vec(), vec![batch_dim, out_dim]);
    }

    #[test]
    fn binarize_weight() {
        let weight = Tensor::from_vec(vec![-0.1f32, 1.9f32, 0.0f32, 0.5f32], (2, 2), DEVICE).unwrap();
        let config = BitLinear1_58Config::default();
        let bitlinear = BitLinear1_58::new(weight.clone(), None, config);
        let binarized_weight = bitlinear.binarize_weight().unwrap();

        assert_eq!(binarized_weight.to_vec2::<f32>().unwrap(), vec![vec![-0f32, 1f32], vec![0f32, -1f32]]);
    }
}