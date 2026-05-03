# llm-serving-stack
 
A from-scratch LLM inference serving platform implementing disaggregated prefill/decode. Built in Rust and Go.
 
# What it is
 
Most LLM serving systems run prefill and decode on the same GPU. This system separates them prefill workers process prompts (compute-bound), decode workers generate tokens (memory-bound), and KV cache blocks transfer between them over gRPC. Every component is built from scratch: the transformer, KV cache, database, and scheduler.
 
## Components
 
**Transformer** (`rust/transformer`) — LLaMA 3.1 compatible inference engine. RMSNorm, RoPE, Grouped Query Attention, SwiGLU FFN. Sparse attention patterns. Custom CUDA kernel for batched attention. Loads pretrained weights via safetensors.
 
**KV Cache** (`rust/kvcache`) — PagedAttention block allocator. Fixed-size blocks eliminate memory fragmentation. Block transfer protocol for disaggregated inference. LRU eviction when GPU memory is under pressure.
 
**Database** (`rust/db`) — Append-only time-series DB. WAL with group commit, lock-free skiplist memtable, SSTable with bloom filters, size-tiered compaction with TTL, B-tree time-range index. ~20k writes/sec, 427ns point reads.
 
**Scheduler** (`go/scheduler`) — GPU-aware HTTP backend. Two-phase routing: selects prefill worker by GPU utilization, decode worker by KV cache availability. Prometheus metrics, Kubernetes autoscaling.
