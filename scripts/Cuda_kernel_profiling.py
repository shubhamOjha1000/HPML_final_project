import torch
import time
from collections import defaultdict
from contextlib import contextmanager
from torch.profiler import profile, record_function, ProfilerActivity
import json

# ============================================================================
#                  ENHANCED PROFILER WITH CHROME TRACE SUPPORT
# ============================================================================

class GPT2Profiler:
    """
    Enhanced profiler for GPT-2 model with Chrome Trace support.
    Tracks time per component and captures ATen ops + CUDA kernels.
    """
    def __init__(self):
        self.timings = defaultdict(list)
        self.enabled = True
        self.torch_profiler = None

    @contextmanager
    def profile(self, name):
        """Context manager for timing code blocks."""
        if not self.enabled:
            yield
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            self.timings[name].append(elapsed)

    def reset(self):
        """Clear all timing data."""
        self.timings.clear()

    def report(self, title="Profiling Report", base_component="Total_Forward_Pass"):
        """Print a formatted report of all timings."""
        if not self.timings:
            print("No profiling data collected.")
            return

        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80)

        if base_component in self.timings:
            base_time = sum(self.timings[base_component])
            print(f"Percentages relative to: {base_component}")
        else:
            base_time = sum(sum(times) for times in self.timings.values())
            print(f"Percentages relative to: All measurements")

        print(f"{'Component':<35} {'Avg (ms)':>12} {'Total (ms)':>14} {'Count':>8} {'%':>7}")
        print("-"*80)

        gpt2_components = [
            "Total_Forward_Pass",
            "Embeddings",
            "All_Transformer_Blocks",
            "Final_LayerNorm",
            "LM_Head_Projection"
        ]

        sorted_timings = sorted(
            self.timings.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )

        for name, times in sorted_timings:
            if name.startswith("Layer_") or name.startswith("Block_") or name.startswith("Attn_") or name.startswith("MLP_"):
                continue

            if name not in gpt2_components:
                continue

            avg_ms = (sum(times) / len(times)) * 1000
            total_ms = sum(times) * 1000
            count = len(times)
            percentage = (sum(times) / base_time * 100) if base_time > 0 else 0

            print(f"{name:<35} {avg_ms:>12.3f} {total_ms:>14.3f} {count:>8} {percentage:>6.1f}%")

        print("-"*80)
        print(f"BASE: {base_component:<24} {'':<12} {base_time*1000:>14.3f} {'':<8} {'100.0%':>7}")
        print("="*80 + "\n")

    def report_block_level(self, title="Block-Level Profiling"):
        """Print a separate report for Block-level components."""
        if not self.timings:
            print("No profiling data collected.")
            return

        block_components = [
            "Block_LayerNorm_1",
            "Block_Attention",
            "Block_LayerNorm_2",
            "Block_MLP"
        ]

        has_block_data = any(comp in self.timings for comp in block_components)
        if not has_block_data:
            return

        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80)

        if "All_Transformer_Blocks" in self.timings:
            base_time = sum(self.timings["All_Transformer_Blocks"])
            print(f"Percentages relative to: All_Transformer_Blocks")
        else:
            print("Warning: All_Transformer_Blocks not found")
            return

        print(f"{'Component':<35} {'Avg (ms)':>12} {'Total (ms)':>14} {'Count':>8} {'%':>7}")
        print("-"*80)

        block_timings = [(name, times) for name, times in self.timings.items()
                        if name in block_components]
        block_timings.sort(key=lambda x: sum(x[1]), reverse=True)

        for name, times in block_timings:
            avg_ms = (sum(times) / len(times)) * 1000
            total_ms = sum(times) * 1000
            count = len(times)
            percentage = (sum(times) / base_time * 100) if base_time > 0 else 0

            display_name = name.replace("Block_", "")
            if display_name == "LayerNorm_1":
                display_name = "LayerNorm (pre-attention)"
            elif display_name == "LayerNorm_2":
                display_name = "LayerNorm (pre-MLP)"
            elif display_name == "Attention":
                display_name = "Multi-Head Attention"
            elif display_name == "MLP":
                display_name = "MLP (Feed-Forward)"

            print(f"{display_name:<35} {avg_ms:>12.3f} {total_ms:>14.3f} {count:>8} {percentage:>6.1f}%")

        print("-"*80)
        avg_blocks = (base_time / len(self.timings["All_Transformer_Blocks"])) * 1000
        print(f"BASE: All_Transformer_Blocks {'':<7} {avg_blocks:>12.3f} {base_time*1000:>14.3f} {'':<8} {'100.0%':>7}")
        print("="*80 + "\n")

    def get_average(self, name):
        """Get average time for a specific component."""
        if name not in self.timings or not self.timings[name]:
            return 0.0
        return sum(self.timings[name]) / len(self.timings[name])


# Global profiler instance
profiler = GPT2Profiler()


# ============================================================================
#                     PROFILED ATTENTION FORWARD PASS
# ============================================================================

def profiled_attention_forward(attn_module, x):
    """Profiled forward pass for CausalMultiHeadSelfAttention."""
    with record_function("ATTENTION_BLOCK"):
        B, T, C = x.size()

        # QKV Projection
        with record_function("QKV_Projection"):
            with profiler.profile("Attn_QKV_Projection"):
                k = attn_module.W_key(x)
                q = attn_module.W_query(x)
                v = attn_module.W_value(x)

        # Reshape for multi-head attention
        with record_function("QKV_Reshape"):
            with profiler.profile("Attn_QKV_Reshape"):
                k = k.view(B, T, attn_module.n_head, C // attn_module.n_head).transpose(1, 2)
                q = q.view(B, T, attn_module.n_head, C // attn_module.n_head).transpose(1, 2)
                v = v.view(B, T, attn_module.n_head, C // attn_module.n_head).transpose(1, 2)

        # Compute attention scores
        with record_function("Attention_Scores"):
            with profiler.profile("Attn_Scores_Compute"):
                att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))

        # Apply causal mask
        with record_function("Causal_Mask"):
            with profiler.profile("Attn_Mask_Apply"):
                att = att.masked_fill(attn_module.bias[:, :, :T, :T] == 0, float('-inf'))

        # Softmax and dropout
        with record_function("Softmax_Dropout"):
            with profiler.profile("Attn_Softmax"):
                att = torch.nn.functional.softmax(att, dim=-1)
                att = attn_module.attn_dropout(att)

        # Apply attention to values
        with record_function("Apply_Attention"):
            with profiler.profile("Attn_Values_Apply"):
                y = att @ v

        # Reshape output
        with record_function("Output_Reshape"):
            with profiler.profile("Attn_Output_Reshape"):
                y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        with record_function("Output_Projection"):
            with profiler.profile("Attn_Output_Projection"):
                y = attn_module.c_proj(y)

    return y


# ============================================================================
#                     PROFILED MLP FORWARD PASS
# ============================================================================

def profiled_mlp_forward(mlp_module, x):
    """Profiled forward pass for MLP (Feed-Forward Network)."""
    with record_function("MLP_BLOCK"):
        # Expansion layer
        with record_function("MLP_Expansion"):
            with profiler.profile("MLP_Expansion"):
                x = mlp_module.c_fc(x)

        # Activation function
        with record_function("MLP_GELU"):
            with profiler.profile("MLP_Activation"):
                x = mlp_module.act(x)

        # Projection layer
        with record_function("MLP_Projection"):
            with profiler.profile("MLP_Projection"):
                x = mlp_module.c_proj(x)

    return x


# ============================================================================
#                     PROFILED FORWARD PASS WRAPPER
# ============================================================================

def profiled_forward(model, idx):
    """Wrapper around model forward pass with detailed profiling."""
    with record_function("FORWARD_PASS"):
        with profiler.profile("Total_Forward_Pass"):
            device = idx.device
            b, t = idx.size()

            # Embeddings
            with record_function("EMBEDDINGS"):
                with profiler.profile("Embeddings"):
                    pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
                    tok_emb = model.wte(idx)
                    pos_emb = model.wpe(pos)
                    x = model.drop(tok_emb + pos_emb)

            # Transformer blocks
            with record_function("TRANSFORMER_BLOCKS"):
                with profiler.profile("All_Transformer_Blocks"):
                    for layer_idx, block in enumerate(model.h):
                        with record_function(f"LAYER_{layer_idx}"):
                            # Attention path
                            shortcut = x

                            with record_function(f"LayerNorm_1"):
                                with profiler.profile("Block_LayerNorm_1"):
                                    x = block.ln_1(x)

                            with profiler.profile("Block_Attention"):
                                x = profiled_attention_forward(block.attn, x)

                            with record_function("Dropout_Residual_1"):
                                x = block.drop_shortcut(x)
                                x = x + shortcut

                            # MLP path
                            shortcut = x

                            with record_function(f"LayerNorm_2"):
                                with profiler.profile("Block_LayerNorm_2"):
                                    x = block.ln_2(x)

                            with profiler.profile("Block_MLP"):
                                x = profiled_mlp_forward(block.mlp, x)

                            with record_function("Dropout_Residual_2"):
                                x = block.drop_shortcut(x)
                                x = x + shortcut

            # Final layers
            with record_function("FINAL_LAYERNORM"):
                with profiler.profile("Final_LayerNorm"):
                    x = model.ln_f(x)

            with record_function("LM_HEAD"):
                with profiler.profile("LM_Head_Projection"):
                    logits = model.lm_head(x)

    return logits


# ============================================================================
#              MAIN PROFILING FUNCTION WITH CHROME TRACE
# ============================================================================

def profile_gpt2_with_chrome_trace(model, tokenizer, prompt="My name is Shubham Ojha",
                                   num_warmup=5, num_profiled_runs=3, max_tokens=20,
                                   output_file="gpt2_trace.json"):
    """
    Profile GPT-2 model and generate Chrome trace visualization.
    
    Args:
        model: Your GPT-2 model
        tokenizer: GPT-2 tokenizer
        prompt: Input text prompt
        num_warmup: Number of warmup iterations
        num_profiled_runs: Number of profiled forward passes
        max_tokens: Number of tokens to generate
        output_file: Output filename for Chrome trace
    """
    model.eval()
    device = next(model.parameters()).device

    print("\n" + "="*80)
    print("  GPT-2 PROFILING WITH CHROME TRACE")
    print("="*80)
    print(f"Device: {device}")
    print(f"Prompt: '{prompt}'")
    print(f"Warmup runs: {num_warmup}")
    print(f"Profiled runs: {num_profiled_runs}")
    print(f"Max tokens: {max_tokens}")
    print(f"Output file: {output_file}")
    print("="*80 + "\n")

    # Reset profiler
    profiler.reset()

    # ========== WARMUP ==========
    print("🔥 Warming up GPU...")
    with torch.no_grad():
        for _ in range(num_warmup):
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            for _ in range(min(5, max_tokens)):
                logits = model(input_ids)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)

    print("✓ Warmup complete\n")

    # ========== PROFILING WITH TORCH PROFILER ==========
    print(f"📊 Starting PyTorch profiler for {num_profiled_runs} runs...")
    
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with torch.no_grad():
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
            
            for run_idx in range(num_profiled_runs):
                input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
                
                for token_idx in range(max_tokens):
                    logits = profiled_forward(model, input_ids)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                
                # Step the profiler
                prof.step()
                
                if (run_idx + 1) % 1 == 0:
                    print(f"  ✓ Completed run {run_idx + 1}/{num_profiled_runs}")

    print(f"✓ Profiling complete!\n")

    # ========== EXPORT CHROME TRACE ==========
    print(f"💾 Exporting Chrome trace to {output_file}...")
    prof.export_chrome_trace(output_file)
    print(f"✓ Chrome trace saved to {output_file}\n")

    # ========== PRINT REPORTS ==========
    
    # High-level report
    profiler.report(f"GPT-2 High-Level Profiling ({num_profiled_runs} runs)")
    profiler.report_block_level(f"Block-Level Breakdown ({num_profiled_runs} runs)")

    # ========== PYTORCH PROFILER SUMMARY ==========
    print("\n" + "="*80)
    print("  PYTORCH PROFILER - TOP CPU OPERATIONS")
    print("="*80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    if torch.cuda.is_available():
        print("\n" + "="*80)
        print("  PYTORCH PROFILER - TOP CUDA OPERATIONS")
        print("="*80)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("\n" + "="*80)
    print("  PYTORCH PROFILER - OPERATIONS BY NAME")
    print("="*80)
    print(prof.key_averages(group_by_input_shape=False).table(
        sort_by="cpu_time_total", row_limit=30))

    # ========== INSTRUCTIONS ==========
    print("\n" + "="*80)
    print("  HOW TO VIEW CHROME TRACE")
    print("="*80)
    print("1. Open Google Chrome browser")
    print("2. Navigate to: chrome://tracing")
    print(f"3. Click 'Load' and select: {output_file}")
    print("4. Use WASD keys to navigate:")
    print("   - W/S: Zoom in/out")
    print("   - A/D: Pan left/right")
    print("   - Click on operations to see details")
    print("="*80 + "\n")

    print("✅ Profiling complete! Check the Chrome trace for detailed visualization.\n")

    return prof


# ============================================================================
#                              RUN PROFILING
# ============================================================================

if __name__ == "__main__":
    from transformers import GPT2Tokenizer
    from GPT2_124M_model_code import GPT2
    from utils import load_pretrained_weights

    # Configuration
    GPT2_CONFIG_124M = {
        'vocab_size': 50257,
        'n_positions': 1024,
        'n_embd': 768,
        'n_layer': 12,
        'n_head': 12,
        'n_inner': 3072,
        'activation': 'gelu',
        'resid_pdrop': 0.1,
        'embd_pdrop': 0.1,
        'attn_pdrop': 0.1,
        'layer_norm_epsilon': 1e-5
    }

    print("\n📚 Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    print("🔧 Creating GPT2 model...")
    model = GPT2(GPT2_CONFIG_124M)
    
    print("⬇️  Loading pretrained weights...")
    model = load_pretrained_weights(model)
    model.eval()

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"✓ Model loaded on {device}\n")

    # Run profiling with Chrome trace
    prof = profile_gpt2_with_chrome_trace(
        model=model,
        tokenizer=tokenizer,
        prompt="What is your name?",
        num_warmup=5,
        num_profiled_runs=3,
        max_tokens=20,
        output_file="gpt2_detailed_trace.json"
    )