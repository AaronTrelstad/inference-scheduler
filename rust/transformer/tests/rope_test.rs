use transformer::rope::RoPE;
use transformer::config::LlamaConfig;
use candle_core::{Device, DType, Tensor};

fn test_config() -> LlamaConfig {
    LlamaConfig {
        vocab_size:                  128256,
        hidden_size:                 512,    
        n_layers:                    2,
        n_heads:                     8,
        n_kv_heads:                  2,
        intermediate_size:           1024,
        max_seq_len:                 128,    
        rope_theta:                  500000.0,
        rope_scaling_factor:         8.0,
        rope_low_freq_factor:        1.0,
        rope_high_freq_factor:       4.0,
        rope_original_max_seq_len:   8192,
        rms_norm_eps:                1e-5,
        bos_token_id:                128000,
        eos_token_id:                128001,
        head_dim:                    512 / 8,  
        n_rep:                       8 / 2,    
    }
}

#[test]
fn test_rope_table_shape() {
    let device = Device::Cpu;
    let config = test_config();
    let rope = RoPE::new(&config, &device).unwrap();

    // cos/sin tables should be [max_seq_len, head_dim/2]
    assert_eq!(rope.cos_shape(), &[config.max_seq_len, config.head_dim / 2]);
    assert_eq!(rope.sin_shape(), &[config.max_seq_len, config.head_dim / 2]);
}

#[test]
fn test_rope_position_zero_is_identity() {
    let device = Device::Cpu;
    let config = test_config();
    let rope   = RoPE::new(&config, &device).unwrap();

    let x = Tensor::randn(0f32, 1f32, (1, config.n_heads, 1, config.head_dim), &device).unwrap();
    let out = rope.apply(&x, 0).unwrap();

    let x_data   = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let out_data = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    for (a, b) in x_data.iter().zip(out_data.iter()) {
        assert!((a - b).abs() < 1e-5, "at pos=0 output should equal input, got {} vs {}", a, b);
    }
}

#[test]
fn test_rope_output_shape_preserved() {
    let device = Device::Cpu;
    let config = test_config();
    let rope   = RoPE::new(&config, &device).unwrap();

    let x = Tensor::randn(0f32, 1f32, (1, config.n_heads, 4, config.head_dim), &device).unwrap();
    let out = rope.apply(&x, 0).unwrap();

    assert_eq!(out.dims(), x.dims());
}

#[test]
fn test_rope_different_positions_give_different_output() {
    let device = Device::Cpu;
    let config = test_config();
    let rope   = RoPE::new(&config, &device).unwrap();

    let x = Tensor::ones((1, config.n_heads, 1, config.head_dim), DType::F32, &device).unwrap();

    let out_pos0 = rope.apply(&x, 0).unwrap();
    let out_pos5 = rope.apply(&x, 5).unwrap();

    let data0 = out_pos0.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let data5 = out_pos5.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let any_different = data0.iter().zip(data5.iter()).any(|(a, b)| (a - b).abs() > 1e-5);
    assert!(any_different, "different positions should give different rotations");
}

#[test]
fn test_rope_norm_preserved() {
    let device = Device::Cpu;
    let config = test_config();
    let rope   = RoPE::new(&config, &device).unwrap();

    let x   = Tensor::randn(0f32, 1f32, (1, config.n_heads, 1, config.head_dim), &device).unwrap();
    let out = rope.apply(&x, 3).unwrap();

    let x_norm = x.sqr().unwrap()
                  .sum_all().unwrap()
                  .sqrt().unwrap()
                  .to_scalar::<f32>().unwrap();

    let out_norm = out.sqr().unwrap()
                      .sum_all().unwrap()
                      .sqrt().unwrap()
                      .to_scalar::<f32>().unwrap();

    assert!(
        (x_norm - out_norm).abs() < 1e-3,
        "rotation should preserve norm: {} vs {}", x_norm, out_norm
    );
}

#[test]
fn test_rope_relative_position_encoding() {
    let device = Device::Cpu;
    let config = test_config();
    let rope   = RoPE::new(&config, &device).unwrap();

    let q = Tensor::randn(0f32, 1f32, (1, 1, 1, config.head_dim), &device).unwrap();
    let k = Tensor::randn(0f32, 1f32, (1, 1, 1, config.head_dim), &device).unwrap();

    let q5  = rope.apply(&q, 5).unwrap();
    let k3  = rope.apply(&k, 3).unwrap();
    let q10 = rope.apply(&q, 10).unwrap();
    let k8  = rope.apply(&k, 8).unwrap();

    let dot1 = q5.flatten_all().unwrap()
                 .mul(&k3.flatten_all().unwrap()).unwrap()
                 .sum_all().unwrap()
                 .to_scalar::<f32>().unwrap();

    let dot2 = q10.flatten_all().unwrap()
                  .mul(&k8.flatten_all().unwrap()).unwrap()
                  .sum_all().unwrap()
                  .to_scalar::<f32>().unwrap();

    assert!(
        (dot1 - dot2).abs() < 1e-3,
        "relative position encoding failed: {} vs {}", dot1, dot2
    );
}
