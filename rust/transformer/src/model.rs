use crate::attention::KVCache;
use crate::block::LlamaBlock;
use crate::config::LlamaConfig;
use crate::rmsnorm::RMSNorm;
use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Linear, Module};

pub struct Llama {
    embedding: Embedding,
    blocks: Vec<LlamaBlock>,
    norm: RMSNorm,
    lm_head: Linear,
    config: LlamaConfig,
}

impl Llama {
    pub fn new(
        embedding: Embedding,
        blocks:    Vec<LlamaBlock>,
        norm:      RMSNorm,
        lm_head:   Linear,
        config:    LlamaConfig,
    ) -> Self {
        Self {
            embedding,
            blocks,
            norm,
            lm_head,
            config,
        }
    }

    pub fn forward(
        &self,
        tokens: &Tensor,
        kv_caches: &mut Vec<Option<KVCache>>,
        pos: usize,
    ) -> Result<Tensor> {
        let mut x = self.embedding.forward(tokens)?;

        for (block, kv_cache) in self.blocks.iter().zip(kv_caches.iter_mut()) {
            x = block.forward(&x, kv_cache, pos)?;
        }
        
        let x = self.norm.forward(&x)?;
        let x = x.narrow(1, x.dim(1)? - 1, 1)?;  
        self.lm_head.forward(&x)
    }

    pub fn empty_kv_caches(&self) -> Vec<Option<KVCache>> {
        vec![None; self.config.n_layers]
    }
}
