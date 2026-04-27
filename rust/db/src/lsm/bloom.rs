use siphasher::sip::SipHasher13;
use std::hash::{Hash, Hasher};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct BloomFilter {
    bits: Vec<u8>,
    num_bits: usize,
    num_hashes: usize 
}

impl BloomFilter {
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let num_bits = (-(expected_items as f64) * false_positive_rate.ln()) / (2.0_f64.ln().powi(2));
        let num_bits = num_bits as usize;
    
        let num_hashes = ((num_bits as f64 / expected_items as f64) * 2.0_f64.ln()) as usize;
    
        Self {
            bits: vec![0u8; (num_bits + 7) / 8],
            num_bits,
            num_hashes,
        }
    }

    pub fn insert(&mut self, key: &[u8]) {
        for i in 0..self.num_hashes {
            let bit = Self::hash_function(key, i) % self.num_bits;
            let byte_index = bit / 8;
            let bit_offset = bit % 8;
            self.bits[byte_index] |= 1 << bit_offset;
        }
    }

    pub fn contains(&self, key: &[u8]) -> bool {
        for i in 0..self.num_hashes {
            let bit = Self::hash_function(key, i) % self.num_bits;
            let byte_index = bit / 8;
            let bit_offset = bit % 8;
            if (self.bits[byte_index] >> bit_offset) & 1 != 1 {
                return false;
            }
        }

        true
    }

    fn hash_function(key: &[u8], seed: usize) -> usize {
        let mut hasher = SipHasher13::new_with_keys(seed as u64, 0);
        key.hash(&mut hasher);
        hasher.finish() as usize
    }
}
