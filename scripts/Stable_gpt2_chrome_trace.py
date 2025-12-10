import torch
import torch.nn as nn
import json
import time
from contextlib import contextmanager
from typing import List, Dict, Any
from GPT2_124M_model_code import GPT2
from transformers import GPT2Tokenizer
from utils import load_pretrained_weights


class ChromeTraceProfiler:
    """
    Generate beautiful Chrome trace profiles for GPT2 model inference with GPU support.
    View the trace by opening chrome://tracing in Chrome and loading the JSON file.
    """
    
    def __init__(self, device='cuda'):
        self.events: List[Dict[str, Any]] = []
        self.start_time = None
        self.device = device
        self.use_cuda = device.type == 'cuda' if isinstance(device, torch.device) else 'cuda' in str(device)
        
        # Use Chrome tracing's predefined color names
        self.module_colors = {
            'embedding': 'thread_state_running',        # Red
            'attention': 'cq_build_running',            # Teal/Cyan
            'mlp': 'rail_idle',                         # Blue
            'layernorm': 'rail_response',               # Orange
            'block': 'good',                            # Green
            'forward': 'rail_load',                     # Yellow
            'sampling': 'generic_work',                 # Purple
            'total': 'rail_animation',                  # Light Blue
            'cuda': 'thread_state_iowait',              # Dark Green
            'memory': 'thread_state_uninterruptible',   # Pink/Magenta
        }
        
    def get_color(self, module_name: str) -> str:
        """Get color for module based on name."""
        module_lower = module_name.lower()
        for key, color in self.module_colors.items():
            if key in module_lower:
                return color
        return 'generic_work'  # Default grey - this is a valid Chrome tracing color
    
    def reset(self):
        """Reset the profiler for a new trace."""
        self.events = []
        if self.use_cuda:
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
    
    def _get_timestamp(self) -> float:
        """Get current timestamp, syncing GPU if needed."""
        if self.use_cuda:
            torch.cuda.synchronize()
        return time.perf_counter()
    
    @contextmanager
    def profile(self, name: str, category: str = "inference", args: Dict = None, tid: int = None):
        """
        Context manager to profile a code block.
        
        Args:
            name: Name of the operation
            category: Category for grouping
            args: Additional arguments to display in trace
            tid: Thread ID (1 for CPU, 2 for GPU compute, 3 for GPU memory)
        """
        if self.start_time is None:
            self.reset()
        
        # Auto-assign thread based on category if not specified
        if tid is None:
            if self.use_cuda and category in ['cuda', 'forward', 'attention', 'mlp', 'embedding', 'layernorm']:
                tid = 2  # GPU compute thread
            else:
                tid = 1  # CPU thread
        
        start = self._get_timestamp()
        
        # Add memory info for GPU
        if self.use_cuda and tid == 2:
            mem_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            mem_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
            if args is None:
                args = {}
            args.update({
                'gpu_mem_allocated_mb': f"{mem_allocated:.2f}",
                'gpu_mem_reserved_mb': f"{mem_reserved:.2f}"
            })
        
        try:
            yield
        finally:
            end = self._get_timestamp()
            duration_us = (end - start) * 1e6  # Convert to microseconds
            timestamp_us = (start - self.start_time) * 1e6
            
            event = {
                'name': name,
                'cat': category,
                'ph': 'X',  # Complete event
                'ts': timestamp_us,
                'dur': duration_us,
                'pid': 0,  # Process ID
                'tid': tid,  # Thread ID
                'args': args or {},
                'cname': self.get_color(name)  # Color name
            }
            
            self.events.append(event)
    
    def add_instant_event(self, name: str, category: str = "marker", tid: int = 1):
        """Add an instant marker event."""
        if self.start_time is None:
            self.reset()
        
        if self.use_cuda:
            torch.cuda.synchronize()
            
        timestamp_us = (time.perf_counter() - self.start_time) * 1e6
        
        event = {
            'name': name,
            'cat': category,
            'ph': 'i',  # Instant event
            'ts': timestamp_us,
            'pid': 0,
            'tid': tid,
            's': 'g',  # Global scope
            'args': {}
        }
        self.events.append(event)
    
    def add_counter_event(self, name: str, value: float, tid: int = 3):
        """Add a counter event for tracking metrics like memory usage."""
        if self.start_time is None:
            self.reset()
            
        if self.use_cuda:
            torch.cuda.synchronize()
            
        timestamp_us = (time.perf_counter() - self.start_time) * 1e6
        
        event = {
            'name': name,
            'ph': 'C',  # Counter event
            'ts': timestamp_us,
            'pid': 0,
            'args': {name: value}
        }
        self.events.append(event)
    
    def track_gpu_memory(self):
        """Track current GPU memory usage as counter events."""
        if self.use_cuda:
            mem_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            mem_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
            
            self.add_counter_event("GPU Memory Allocated (MB)", mem_allocated, tid=3)
            self.add_counter_event("GPU Memory Reserved (MB)", mem_reserved, tid=3)
    
    def save(self, filename: str = "gpt2_trace.json"):
        """Save the trace to a JSON file."""
        # Add thread name metadata
        thread_names = [
            {'name': 'thread_name', 'ph': 'M', 'pid': 0, 'tid': 1, 'args': {'name': 'CPU Operations'}},
            {'name': 'thread_name', 'ph': 'M', 'pid': 0, 'tid': 2, 'args': {'name': 'GPU Compute'}},
            {'name': 'thread_name', 'ph': 'M', 'pid': 0, 'tid': 3, 'args': {'name': 'GPU Memory'}}
        ]
        
        trace_data = {
            'traceEvents': thread_names + self.events,
            'displayTimeUnit': 'ms',
            'metadata': {
                'model': 'GPT2-124M',
                'framework': 'PyTorch',
                'device': str(self.device),
                'cuda_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if self.use_cuda else 'N/A'
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        total_time = sum(e.get('dur', 0) for e in self.events if 'dur' in e) / 1000
        
        print(f"\n{'='*80}")
        print(f"✓ Chrome trace saved to: {filename}")
        print(f"{'='*80}")
        print(f"  📊 Total events: {len(self.events)}")
        print(f"  ⏱️  Total duration: {total_time:.2f} ms")
        if self.use_cuda:
            print(f"  🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"  💾 Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        print(f"{'='*80}")
        print(f"\n🔍 To view the trace:")
        print(f"  1. Open Chrome browser")
        print(f"  2. Go to: chrome://tracing")
        print(f"  3. Click 'Load' and select: {filename}")
        print(f"\n💡 Navigation tips:")
        print(f"  • W/A/S/D - Zoom and pan")
        print(f"  • Click bars for details")
        print(f"  • View separate CPU/GPU timelines")


# Modified GPT2 class with profiling hooks
class GPT2Profiled(GPT2):
    """GPT2 model with built-in profiling support."""
    
    def __init__(self, config, profiler: ChromeTraceProfiler = None):
        super().__init__(config)
        self.profiler = profiler
    
    def forward(self, idx):
        """Forward pass with profiling."""
        device = idx.device
        b, t = idx.size()
        
        with self.profiler.profile(
            "Model Forward Pass", 
            "forward",
            {'batch_size': b, 'seq_length': t},
            tid=2  # GPU compute thread
        ):
            # Track GPU memory at start
            self.profiler.track_gpu_memory()
            
            # Get position indices
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
            
            # Embeddings
            with self.profiler.profile("Token Embedding (wte)", "embedding", tid=2):
                tok_emb = self.wte(idx)
            
            with self.profiler.profile("Position Embedding (wpe)", "embedding", tid=2):
                pos_emb = self.wpe(pos)
            
            with self.profiler.profile("Embedding Dropout", "embedding", tid=2):
                x = self.drop(tok_emb + pos_emb)
            
            # Transformer blocks
            for i, block in enumerate(self.h):
                with self.profiler.profile(f"Block {i}", "block", tid=2):
                    # Layer Norm 1
                    with self.profiler.profile(f"Block {i} - LayerNorm 1", "layernorm", tid=2):
                        shortcut = x
                        x = block.ln_1(x)
                    
                    # Attention
                    with self.profiler.profile(f"Block {i} - Multi-Head Attention", "attention", tid=2):
                        x = block.attn(x)
                    
                    with self.profiler.profile(f"Block {i} - Attention Dropout + Residual", "attention", tid=2):
                        x = block.drop_shortcut(x)
                        x = x + shortcut
                    
                    # Layer Norm 2
                    with self.profiler.profile(f"Block {i} - LayerNorm 2", "layernorm", tid=2):
                        shortcut = x
                        x = block.ln_2(x)
                    
                    # MLP
                    with self.profiler.profile(f"Block {i} - MLP (Feed Forward)", "mlp", tid=2):
                        x = block.mlp(x)
                    
                    with self.profiler.profile(f"Block {i} - MLP Dropout + Residual", "mlp", tid=2):
                        x = block.drop_shortcut(x)
                        x = x + shortcut
                    
                    # Track memory after each block
                    if i % 3 == 0:  # Every 3 blocks
                        self.profiler.track_gpu_memory()
            
            # Final layer norm
            with self.profiler.profile("Final LayerNorm (ln_f)", "layernorm", tid=2):
                x = self.ln_f(x)
            
            # Language model head
            with self.profiler.profile("LM Head Projection", "forward", tid=2):
                logits = self.lm_head(x)
            
            # Track GPU memory at end
            self.profiler.track_gpu_memory()
            
            return logits


def run_profiled_inference(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=50):
    """Run inference with profiling."""
    profiler = model.profiler
    device = next(model.parameters()).device
    use_cuda = device.type == 'cuda'
    
    model.eval()
    
    with profiler.profile("Total Inference", "total", tid=1):
        # Tokenization (CPU)
        with profiler.profile("Tokenization", "preprocessing", tid=1):
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        # Transfer to GPU
        if use_cuda:
            with profiler.profile("Transfer to GPU", "cuda", tid=1):
                input_ids = input_ids.to(device)
                profiler.track_gpu_memory()
        
        original_length = input_ids.shape[1]
        
        # Generation loop
        with profiler.profile(f"Generation Loop ({max_new_tokens} tokens)", "generation", tid=1):
            with torch.no_grad():
                for token_idx in range(max_new_tokens):
                    if input_ids.shape[1] >= model.config['n_positions']:
                        break
                    
                    with profiler.profile(f"Token {token_idx + 1} Generation", "token_gen", tid=1):
                        # Forward pass (this will be profiled internally on GPU)
                        logits = model(input_ids)
                        
                        # Sampling (on GPU)
                        with profiler.profile(f"Token {token_idx + 1} - Sampling", "sampling", tid=2):
                            logits = logits[:, -1, :] / temperature
                            
                            if top_k > 0:
                                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                                logits_filtered = torch.full_like(logits, float('-inf'))
                                logits_filtered.scatter_(1, top_k_indices, top_k_logits)
                                logits = logits_filtered
                            
                            probs = torch.nn.functional.softmax(logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                            input_ids = torch.cat([input_ids, next_token], dim=1)
                            
                            if next_token.item() == tokenizer.eos_token_id:
                                break
                        
                        # Track memory periodically
                        if token_idx % 10 == 0:
                            profiler.track_gpu_memory()
        
        # Transfer back to CPU for decoding
        if use_cuda:
            with profiler.profile("Transfer to CPU", "cuda", tid=1):
                input_ids_cpu = input_ids.cpu()
        else:
            input_ids_cpu = input_ids
        
        # Decoding (CPU)
        with profiler.profile("Detokenization", "postprocessing", tid=1):
            generated_text = tokenizer.decode(input_ids_cpu[0], skip_special_tokens=True)
    
    return generated_text


def main():
    print("=" * 80)
    print("🚀 GPT2-124M Chrome Trace Profiler with GPU Support")
    print("=" * 80)
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Device Count: {torch.cuda.device_count()}")
    else:
        device = torch.device('cpu')
        print("\n⚠️  GPU not available, using CPU")
    
    # Initialize profiler
    profiler = ChromeTraceProfiler(device=device)
    profiler.reset()
    
    # Load tokenizer
    print("\n📚 Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Configuration Dictionary for GPT2 124M Model :-
    GPT2_CONFIG_124M = {
        'vocab_size': 50257,      # Vocabulary size
        'n_positions': 1024,       # Maximum sequence length (context window)
        'n_embd': 768,            # Embedding dimension (hidden size)
        'n_layer': 12,            # Number of transformer blocks
        'n_head': 12,             # Number of attention heads
        'n_inner': 3072,          # Inner dimension of feed-forward network (4 * n_embd)
        'activation': 'gelu',     # Activation function
        'resid_pdrop': 0.1,       # Residual dropout probability
        'embd_pdrop': 0.1,        # Embedding dropout probability
        'attn_pdrop': 0.1,        # Attention dropout probability
        'layer_norm_epsilon': 1e-5  # Layer normalization epsilon
    }

    
    # Create profiled model
    print("🔧 Creating GPT2 model with profiling...")
    model = GPT2Profiled(GPT2_CONFIG_124M, profiler=profiler)
    
    # Load pretrained weights
    model = load_pretrained_weights(model)
    model.eval()
    model = model.to(device)
    
    # Warm up GPU
    if device.type == 'cuda':
        print("🔥 Warming up GPU...")
        dummy_input = torch.randint(0, 50257, (1, 10)).to(device)
        with torch.no_grad():
            _ = model(dummy_input)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Run profiled inference
    prompt = "The future of artificial intelligence is"
    max_tokens = 30
    
    print(f"\n{'─' * 80}")
    print(f"📝 Prompt: '{prompt}'")
    print(f"🎯 Generating {max_tokens} tokens...")
    print(f"{'─' * 80}\n")
    
    profiler.add_instant_event("⚡ Inference Start", tid=1)
    
    generated_text = run_profiled_inference(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=0.8,
        top_k=50
    )
    
    profiler.add_instant_event("✓ Inference Complete", tid=1)
    
    print("\n" + "=" * 80)
    print("📄 Generated Text:")
    print("=" * 80)
    print(generated_text)
    print("=" * 80)
    
    # Save trace
    profiler.save("gpt2_gpu_trace.json")
    
    print("\n🎨 Trace Visualization Guide:")
    print("  • Red/Salmon: Embeddings & LayerNorm")
    print("  • Teal: Attention mechanisms")
    print("  • Blue: MLP/Feed-forward layers")
    print("  • Mint: Complete transformer blocks")
    print("  • Yellow: Forward pass & projections")
    print("  • Purple: Sampling operations")
    print("  • Sky Blue: Total inference time")
    print("  • Green: CUDA operations")
    print("\n📊 Trace Layout:")
    print("  • Timeline 1: CPU Operations")
    print("  • Timeline 2: GPU Compute")
    print("  • Timeline 3: GPU Memory Usage (counters)")


if __name__ == "__main__":
    main()