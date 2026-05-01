use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Linear};
use transformer::attention::GroupedQueryAttention;
use transformer::block::LlamaBlock;
use transformer::config::LlamaConfig;
use transformer::ffn::SwiGLU;
use transformer::model::Llama;
use transformer::rmsnorm::RMSNorm;
use transformer::rope::RoPE;

fn small_config() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 1024,
        hidden_size: 256,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        intermediate_size: 512,
        max_seq_len: 128,
        rope_theta: 500000.0,
        rope_scaling_factor: 8.0,
        rope_low_freq_factor: 1.0,
        rope_high_freq_factor: 4.0,
        rope_original_max_seq_len: 8192,
        rms_norm_eps: 1e-5,
        bos_token_id: 1,
        eos_token_id: 2,
        head_dim: 256 / 4,
        n_rep: 4 / 2,
    }
}

fn make_block(config: &LlamaConfig, device: &Device) -> LlamaBlock {
    let h = config.hidden_size;
    let qd = config.n_heads * config.head_dim;
    let kd = config.n_kv_heads * config.head_dim;
    let int = config.intermediate_size;

    let attn_norm = RMSNorm::new(
        Tensor::ones(h, DType::F32, device).unwrap(),
        config.rms_norm_eps,
    );
    let ffn_norm = RMSNorm::new(
        Tensor::ones(h, DType::F32, device).unwrap(),
        config.rms_norm_eps,
    );
    let q_proj = Linear::new(Tensor::randn(0f32, 0.02, (qd, h), device).unwrap(), None);
    let k_proj = Linear::new(Tensor::randn(0f32, 0.02, (kd, h), device).unwrap(), None);
    let v_proj = Linear::new(Tensor::randn(0f32, 0.02, (kd, h), device).unwrap(), None);
    let o_proj = Linear::new(Tensor::randn(0f32, 0.02, (h, qd), device).unwrap(), None);
    let rope = RoPE::new(config, device).unwrap();
    let attn = GroupedQueryAttention::new(q_proj, k_proj, v_proj, o_proj, rope, config);

    let gate_proj = Linear::new(Tensor::randn(0f32, 0.02, (int, h), device).unwrap(), None);
    let up_proj = Linear::new(Tensor::randn(0f32, 0.02, (int, h), device).unwrap(), None);
    let down_proj = Linear::new(Tensor::randn(0f32, 0.02, (h, int), device).unwrap(), None);
    let ffn = SwiGLU::new(gate_proj, up_proj, down_proj);

    LlamaBlock::new(attn_norm, attn, ffn_norm, ffn)
}

fn make_model(config: &LlamaConfig, device: &Device) -> Llama {
    let embedding = Embedding::new(
        Tensor::randn(0f32, 0.02, (config.vocab_size, config.hidden_size), device).unwrap(),
        config.hidden_size,
    );
    let blocks: Vec<LlamaBlock> = (0..config.n_layers)
        .map(|_| make_block(config, device))
        .collect();
    let norm = RMSNorm::new(
        Tensor::ones(config.hidden_size, DType::F32, device).unwrap(),
        config.rms_norm_eps,
    );
    let lm_head = Linear::new(
        Tensor::randn(0f32, 0.02, (config.vocab_size, config.hidden_size), device).unwrap(),
        None,
    );

    Llama::new(embedding, blocks, norm, lm_head, config.clone())
}

#[test]
fn test_model_prefill_shape() {
    let device = Device::Cpu;
    let config = small_config();
    let model = make_model(&config, &device);

    let tokens = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
    let mut caches = model.empty_kv_caches();
    let logits = model.forward(&tokens, &mut caches, 0).unwrap();

    assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);
}

#[test]
fn test_model_decode_shape() {
    let device = Device::Cpu;
    let config = small_config();
    let model = make_model(&config, &device);

    let tokens = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
    let mut caches = model.empty_kv_caches();
    model.forward(&tokens, &mut caches, 0).unwrap();

    let next_token = Tensor::from_vec(vec![5u32], (1, 1), &device).unwrap();
    let logits = model.forward(&next_token, &mut caches, 4).unwrap();

    assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);
}

#[test]
fn test_model_logits_finite() {
    let device = Device::Cpu;
    let config = small_config();
    let model = make_model(&config, &device);

    let tokens = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &device).unwrap();
    let mut caches = model.empty_kv_caches();
    let logits = model.forward(&tokens, &mut caches, 0).unwrap();
    let data = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    assert!(
        data.iter().all(|v| v.is_finite()),
        "logits contain NaN or inf"
    );
}

#[test]
fn test_model_kv_caches_populated() {
    let device = Device::Cpu;
    let config = small_config();
    let model = make_model(&config, &device);

    let tokens = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
    let mut caches = model.empty_kv_caches();
    model.forward(&tokens, &mut caches, 0).unwrap();

    assert!(
        caches.iter().all(|c| c.is_some()),
        "all KV caches should be populated"
    );

    for cache in &caches {
        let k_len = cache.as_ref().unwrap().k.dim(2).unwrap();
        assert_eq!(k_len, 4, "each cache should have 4 KV entries");
    }
}

#[test]
fn test_model_autoregressive_generation() {
    let device = Device::Cpu;
    let config = small_config();
    let model = make_model(&config, &device);

    let prompt = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &device).unwrap();
    let mut caches = model.empty_kv_caches();

    let logits = model.forward(&prompt, &mut caches, 0).unwrap();
    let next_token = logits
        .squeeze(0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .argmax(0)
        .unwrap()
        .to_scalar::<u32>()
        .unwrap();

    assert!(
        next_token < config.vocab_size as u32,
        "generated token should be in vocab"
    );

    let token_tensor = Tensor::from_vec(vec![next_token], (1, 1), &device).unwrap();
    let logits2 = model.forward(&token_tensor, &mut caches, 3).unwrap();
    let next_token2 = logits2
        .squeeze(0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .argmax(0)
        .unwrap()
        .to_scalar::<u32>()
        .unwrap();

    assert!(
        next_token2 < config.vocab_size as u32,
        "second generated token should be in vocab"
    );

    let k_len = caches[0].as_ref().unwrap().k.dim(2).unwrap();
    assert_eq!(k_len, 4, "cache should have 3 prompt + 1 decode token");
}

#[test]
fn test_empty_kv_caches_length() {
    let device = Device::Cpu;
    let config = small_config();
    let model = make_model(&config, &device);

    let caches = model.empty_kv_caches();
    assert_eq!(
        caches.len(),
        config.n_layers,
        "should have one cache per layer"
    );
    assert!(
        caches.iter().all(|c| c.is_none()),
        "all caches should start as None"
    );
}
