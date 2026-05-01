use crate::config::LlamaConfig;
use candle_core::{D, DType, Device, Result, Tensor};

pub struct RoPE {
    sin: Tensor,
    cos: Tensor,
}

impl RoPE {
    pub fn cos_shape(&self) -> &[usize] {
        self.cos.dims()
    }

    pub fn sin_shape(&self) -> &[usize] {
        self.sin.dims()
    }

    pub fn new(config: &LlamaConfig, device: &Device) -> Result<Self> {
        let head_dim = config.head_dim;
        let max_seq = config.max_seq_len;
        let theta = config.rope_theta;

        let base_freqs: Vec<f32> = (0..head_dim / 2)
            .map(|i| theta.powf(-2.0 * i as f64 / head_dim as f64) as f32)
            .collect();

        let mut angles: Vec<f32> = Vec::with_capacity(max_seq * head_dim / 2);

        for pos in 0..max_seq {
            for freq in &base_freqs {
                angles.push(pos as f32 * freq);
            }
        }

        let angles = Tensor::from_vec(angles, (max_seq, head_dim / 2), device)?;

        let cos = angles.cos()?;
        let sin = angles.sin()?;

        Ok(Self { cos, sin })
    }

    pub fn apply(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let seq_len = x.dim(2)?;
        let half = x.dim(D::Minus1)? / 2;

        let cos = self.cos.narrow(0, pos, seq_len)?;
        let sin = self.sin.narrow(0, pos, seq_len)?;

        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;

        let new_x1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let new_x2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

        Tensor::cat(&[&new_x1, &new_x2], D::Minus1)
    }
}
