use candle_core::{D, Result, Tensor};

pub fn sign(x: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let zero = Tensor::zeros_like(&x)?;
    let sign_x = x.gt(&zero)?.to_dtype(dtype)? - x.lt(&zero)?.to_dtype(dtype)? - x.eq(&zero)?.to_dtype(dtype)?;
    sign_x

    // let zero = Tensor::zeros_like(&x)?;
    // let one = Tensor::ones_like(&x)?;
    // let minus_one = one.neg()?;

    // let sign_x = x.gt(&zero)?.where_cond(&one, &minus_one);

    // sign_x
}

pub fn min_all(x: &Tensor) -> Result<Tensor> {
    x.flatten_all()?.min(D::Minus1)
}

pub fn alpha(x: &Tensor) -> Result<Tensor> {
    x.mean_all()
}

pub fn beta(x: &Tensor) -> Result<Tensor> {
    x.abs()?.mean_all()
}

pub fn gamma(x: &Tensor) -> Result<Tensor> {
    x.abs()?.max_keepdim(D::Minus1)
}

pub fn ste(x: &Tensor) -> Result<Tensor> {
    sign(&x)?.sub(&x)?.detach().add(&x)
}

pub fn sub_ln(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean_x = x.mean_all()?;
    let x = x.broadcast_sub(&mean_x)?;
    let norm_x = x.sqr()?.mean_all()?;
    let x = x.broadcast_div(&(norm_x + eps)?.sqrt()?);
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    const DEVICE : &Device = &Device::Cpu;

    #[test]
    fn test_sign() {
        let input = Tensor::from_slice(&[0.5, -0.2, 0.0, -0.1, -0.6, 0.3], (1, 6), DEVICE).unwrap().to_dtype(DType::F32).unwrap();
        let output = sign(&input).unwrap();

        assert_eq!(output.to_vec2::<f32>().unwrap(), vec![vec![1f32, -1f32, -1f32, -1f32, -1f32, 1f32]]);
    }

    #[test]
    fn test_min_all() {
        let input = Tensor::from_slice(&[0.5f32, -0.2f32, 0.0f32, -0.1f32, -0.6f32, 0.3f32], (1, 6), DEVICE).unwrap();
        let output = min_all(&input).unwrap();

        assert_eq!(output.to_vec0::<f32>().unwrap(), -0.6f32);
    }

    #[test]
    fn test_alpha() {
        let input = Tensor::from_slice(&[-1.0, 2.0, -3.0, 4.0], (2, 2), DEVICE).unwrap().to_dtype(DType::F32).unwrap();
        let output = alpha(&input).unwrap();
        assert_eq!(output.to_vec0::<f32>().unwrap(), 0.5f32);
    }

    #[test]
    fn test_beta() {
        let input = Tensor::from_slice(&[-1.0, 2.0, -3.0, 4.0], (2, 2), DEVICE).unwrap().to_dtype(DType::F32).unwrap();
        let output = beta(&input).unwrap();
        assert_eq!(output.to_vec0::<f32>().unwrap(), 2.5f32);
    }

    #[test]
    fn test_gamma() {
        let input = Tensor::from_slice(&[-1.0, 2.0, -3.0, 4.0], (2, 2), DEVICE).unwrap().to_dtype(DType::F32).unwrap();
        let output = gamma(&input).unwrap();
        assert_eq!(output.to_vec2::<f32>().unwrap(), vec![vec![2.0f32], vec![4.0f32]]);

        let input = input.reshape((1, 4)).unwrap();
        let output = gamma(&input).unwrap();
        assert_eq!(output.to_vec2::<f32>().unwrap(), vec![vec![4.0f32]]);

        let input = input.reshape(4).unwrap();
        let output = gamma(&input).unwrap();
        assert_eq!(output.to_vec1::<f32>().unwrap(), vec![4.0f32]);
    }

    #[test]
    fn test_sub_ln() {
        let mean = 0f32;
        let std = 1f32;
        let input = Tensor::randn(mean, std, 1, DEVICE).unwrap();
        let output = sub_ln(&input, 1e-6).unwrap();
        assert_eq!(output.to_vec1::<f32>().unwrap(), vec![0f32]);
    }

    // #[test]
    // fn test_ste() {
    //     todo!("test_ste");
    // }
}

