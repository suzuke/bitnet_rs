use candle_core::{Result, Tensor, D};
use candle_nn::{activation, init, rms_norm, Init, Linear, Module, ModuleT, RmsNorm, VarBuilder};

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
    rms_norm: RmsNorm,
    quant_range: (f64, f64),
    eps: f64,
}

impl BitLinear1_58 {
    pub fn new(weight: Tensor, bias: Option<Tensor>, rms_norm: RmsNorm, bit_width: usize, eps: f64) -> Self {
        let q_b = 2.0_f64.powi(bit_width as i32 - 1);
        let quant_range = (-q_b, q_b-1.0);
        Self {
            weight,
            bias,
            rms_norm,
            quant_range,
            eps,
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    fn weight_quant(&self) -> Result<(Tensor, Tensor)> {
        // scale = 1.0 / w.abs().mean().clamp_(min=1e−5)
        // u = (w * scale).round().clamp_(−1, 1) / scale
        let scale = (1.0 / &self.weight.abs()?.mean_all()?.maximum(self.eps)?)?;
        let u = self.weight.broadcast_mul(&scale)?.round()?.clamp(-1.0, 1.0)?;
        Ok((u, scale))
    }

    fn activation_quant(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // scale = 127.0 / x.abs().max(dim=−1, keepdim=True).values.clamp_(min=1e−5)
        // y = (x * scale).round().clamp_(−128, 127) / scale
        let (quant_lo, quant_up) = self.quant_range;
        let scale = (quant_up / &x.abs()?.max_keepdim(D::Minus1)?.maximum(self.eps)?)?;
        let y = x.broadcast_mul(&scale)?.round()?.clamp(quant_lo, quant_up)?;
        Ok((y, scale))
    }

    fn activation_norm_quant(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // x = RMSNorm(x)
        // scale = 127.0 / x.abs().max(dim=−1, keepdim=True).values.clamp_(min=1e−5)
        //y = (x * scale).round().clamp_(−128, 127)
        // return y, scale

        let x = self.rms_norm.forward(x)?;
        let (quant_lo, quant_up) = self.quant_range;
        let scale = (quant_up / &x.abs()?.max_keepdim(D::Minus1)?.maximum(self.eps)?)?;
        let y = x.broadcast_mul(&scale)?.round()?.clamp(quant_lo, quant_up)?;
        Ok((y, scale))
    }

    fn train(&self, x: &Tensor) -> Result<Tensor> {
        let x_norm = self.rms_norm.forward(x)?;
        let (x_quant, x_scale) = self.activation_quant(&x_norm)?;
        let (w_quant, w_scale) = self.weight_quant()?;
        let x_quant = (&x_norm + (x_quant.broadcast_div(&x_scale)? - &x_norm)?.detach())?;
        let w_quant = (&self.weight + (w_quant.broadcast_div(&w_scale) - &self.weight)?.detach())?;
        let y = Linear::new(w_quant, self.bias.clone()).forward(&x_quant)?;
        Ok(y)
    }

    fn infer(&self, x: &Tensor) -> Result<Tensor> {
        // w = self.weight # a 1.58−bit weight tensor with shape [d, k]
        // w_scale = self.weight_scale # a full−precision weight scale tensor with shape [1] 
        // x_quant, x_scale = activation_norm_quant(x)
        // y = gemm_lowbit_kernel(x_quant, w) / w_scale / x_scale
        let (w_quant, w_scale) = self.weight_quant()?;
        let (x_quant, x_scale) = self.activation_norm_quant(x)?;
        let y = Linear::new(w_quant, self.bias.clone()).forward(&x_quant)?.broadcast_div(&w_scale)?.broadcast_div(&x_scale)?;
        Ok(y)
    }

}

impl ModuleT for BitLinear1_58 {
    fn forward_t(&self, x: &Tensor, train: bool) -> Result<Tensor> {      
        if train {
            self.train(&x)
        } else {
            self.infer(&x)
        }
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
    let rms_norm = rms_norm(in_dim, config.eps, vb.pp("rms_norm"))?;
    Ok(BitLinear1_58::new(weight, bias, rms_norm, config.bit_width, config.eps))
}


#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarMap;

    const DEVICE : &Device = &Device::Cpu;
    const DTYPE : DType = DType::F32;

    #[test]
    fn test_bitlinear1_58() {
        let batch_dim = 1;
        let in_dim = 4;
        let out_dim = 3;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DTYPE, DEVICE);
        let weight = Tensor::ones((out_dim, in_dim), DTYPE, DEVICE).unwrap();
        let bias = Tensor::zeros(out_dim, DTYPE, DEVICE).unwrap();
        let config = BitLinear1_58Config::default();
        let rms_norm = rms_norm(in_dim, config.eps, vb).unwrap();
        let bitlinear = BitLinear1_58::new(weight.clone(), Some(bias), rms_norm, config.bit_width, config.eps);

        let input = Tensor::ones((batch_dim, in_dim), DTYPE, DEVICE).unwrap();
        let output = bitlinear.forward_t(&input, true).unwrap();

        assert_eq!(bitlinear.weight().to_vec2::<f32>().unwrap(), weight.to_vec2::<f32>().unwrap());
        assert_eq!(output.dims().to_vec(), vec![batch_dim, out_dim]);
    }

    #[test]
    fn test_weight_quant() {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DTYPE, DEVICE);
        let weight = Tensor::from_vec(vec![-0.1f32, 1.9, 0.0, 0.5], (2, 2), DEVICE).unwrap();
        // // (0.1+1.9+0.0+0.5)/4 = 0.625, [[-0.1, 1.9], [0.0, 0.5]]/(0.625+1e-5) = [[-0.15999744, 3.03995136],[ 0., 0.7999872 ]]
        let config = BitLinear1_58Config::default();
        let bitlinear = BitLinear1_58::new(
            weight,
            None,
            rms_norm(2, config.eps, vb.pp("rms_norm_1")).unwrap(),
            config.bit_width, config.eps);
        let (quat_w, _) = bitlinear.weight_quant().unwrap();
        // // Round([[-0.15999744, 3.03995136],[ 0., 0.7999872 ]]) = [[0.,  3.],[ 0.,  1.]]
        // //Clip([0.,  3.],[ 0.,  1.]) = [[0.,  1.],[0.,  1.]]
        assert_eq!(quat_w.to_vec2::<f32>().unwrap(), vec![vec![0f32, 1f32], vec![0f32, 1f32]]);

        let rand_weight = Tensor::rand(-100f32, 100f32, (10, 10), DEVICE).unwrap();
        let bitlinear = BitLinear1_58::new(
            rand_weight,
            None,
            rms_norm(10, config.eps, vb.pp("rms_norm_2")).unwrap(),
            config.bit_width, config.eps);
        let (quat_w, _) = bitlinear.weight_quant().unwrap();
        let quat_w = quat_w.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // all elements are -1 or 0 or 1
        assert!(quat_w.iter().all(|x| *x == -1f32 || *x == 0f32 || *x == 1f32));


    }
}