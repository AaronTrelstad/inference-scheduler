use crate::block::BlockPool;
use crate::table::BlockTable;
use anyhow::Result;
use candle_core::Device;
use std::collections::HashMap;

pub struct KVCacheConfig {
    pub n_blocks: usize,
    pub block_size: usize,
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

pub struct KVCacheManager {
    pool: BlockPool,
    block_tables: HashMap<String, BlockTable>,
    config: KVCacheConfig,
}

impl KVCacheManager {
    pub fn new(config: KVCacheConfig, device: &Device) -> Result<Self> {
        let pool = BlockPool::new(
            config.n_blocks,
            config.block_size,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            device,
        )?;

        Ok(Self {
            pool,
            block_tables: HashMap::new(),
            config,
        })
    }

    pub fn allocate(&mut self, job_id: &str, n_tokens: usize) -> Result<()> {
        let mut table = BlockTable::new(job_id.to_string(), self.config.block_size);
        let n_blocks = BlockTable::blocks_needed(n_tokens, self.config.block_size);

        for _ in 0..n_blocks {
            let block_idx = match self.pool.alloc() {
                Some(idx) => idx,
                None => {
                    self.pool.evict_lru();
                    self.pool
                        .alloc()
                        .ok_or_else(|| anyhow::anyhow!("out of kv block"))?
                }
            };

            table.append_block(block_idx);
        }

        self.block_tables.insert(job_id.to_string(), table);
        Ok(())
    }
}
