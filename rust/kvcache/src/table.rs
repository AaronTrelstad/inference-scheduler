pub struct BlockTable {
    pub request_id: String,
    pub blocks: Vec<usize>,
    pub n_tokens: usize,
    pub block_size: usize,
}

impl BlockTable {
    pub fn new(request_id: String, block_size: usize) -> Self {
        Self {
            request_id,
            blocks: Vec::new(),
            n_tokens: 0,
            block_size
        }
    }

    pub fn logical_block(&self, pos: usize) -> usize {
        pos / self.block_size
    }

    pub fn slot_in_block(&self, pos: usize) -> usize {
        pos % self.block_size
    }

    pub fn blocks_needed(n_tokens: usize, block_size: usize) -> usize {
        n_tokens.div_ceil(block_size)
    }

    pub fn last_block_full(&self) -> bool {
        if self.blocks.is_empty() {
            return true;
        }
        self.n_tokens % self.block_size == 0
    }

    pub fn append_block(&mut self, physical_idx: usize) {
        self.blocks.push(physical_idx);
    }

    pub fn physical_block(&self, pos: usize) -> Option<usize> {
        let logical = self.logical_block(pos);
        self.blocks.get(logical).copied()
    }
    
    pub fn increment_tokens(&mut self) {
        self.n_tokens += 1;
    }

    pub fn current_slot(&self) -> usize {
        self.slot_in_block(self.n_tokens)
    }
}
