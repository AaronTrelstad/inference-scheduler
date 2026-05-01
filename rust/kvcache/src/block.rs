use candle_core::{DType, Device, Result, Tensor};

pub struct BlockPool {
    pub n_blocks: usize,
    pub block_size: usize,
    pub blocks: Vec<Block>,
    pub free_list: Vec<usize>,
    pub k_data: Tensor,
    pub v_data: Tensor,
}

pub struct Block {
    pub block_idx: usize,
    pub n_tokens: usize,
    pub ref_count: usize,
    pub last_access: u64,
}

impl BlockPool {
    pub fn new(
        n_blocks: usize,
        block_size: usize,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let k_data = Tensor::zeros(
            (n_blocks, n_layers, n_kv_heads, block_size, head_dim),
            DType::F32,
            device,
        )?;
        let v_data = Tensor::zeros(
            (n_blocks, n_layers, n_kv_heads, block_size, head_dim),
            DType::F32,
            device,
        )?;

        let blocks = (0..n_blocks)
            .map(|i| Block {
                block_idx: i,
                n_tokens: 0,
                ref_count: 0,
                last_access: 0,
            })
            .collect();

        let free_list = (0..n_blocks).collect();

        Ok(Self {
            n_blocks,
            block_size,
            blocks,
            free_list,
            k_data,
            v_data,
        })
    }

    pub fn alloc(&mut self) -> Option<usize> {
        self.free_list.pop()
    }

    pub fn free(&mut self, block_idx: usize) {
        self.blocks[block_idx].n_tokens = 0;
        self.blocks[block_idx].ref_count = 0;
        self.free_list.push(block_idx);
    }

    pub fn n_free(&self) -> usize {
        self.free_list.len()
    }
}
