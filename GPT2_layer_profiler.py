import torch
import time
from collections import defaultdict
from contextlib import contextmanager

# ============================================================================
#                           PROFILER CLASS
# ============================================================================

class GPT2Profiler:
    """
    High-level profiler for GPT-2 model.
    Tracks time per forward pass and time per component.
    """
    def __init__(self):
        self.timings = defaultdict(list)
        self.enabled = True

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

    def report_attention_breakdown(self, title="Attention Mechanism Breakdown"):
        """Print detailed breakdown of attention operations."""
        if not self.timings:
            print("No profiling data collected.")
            return

        attn_components = [
            "Attn_QKV_Projection",
            "Attn_QKV_Reshape",
            "Attn_Scores_Compute",
            "Attn_Mask_Apply",
            "Attn_Softmax",
            "Attn_Values_Apply",
            "Attn_Output_Reshape",
            "Attn_Output_Projection"
        ]

        has_attn_data = any(comp in self.timings for comp in attn_components)
        if not has_attn_data:
            return

        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80)

        if "Block_Attention" in self.timings:
            base_time = sum(self.timings["Block_Attention"])
            print(f"Percentages relative to: Block_Attention")
        else:
            print("Warning: Block_Attention not found")
            return

        print(f"{'Component':<35} {'Avg (ms)':>12} {'Total (ms)':>14} {'Count':>8} {'%':>7}")
        print("-"*80)

        attn_timings = [(name, times) for name, times in self.timings.items()
                       if name in attn_components]
        attn_timings.sort(key=lambda x: sum(x[1]), reverse=True)

        for name, times in attn_timings:
            avg_ms = (sum(times) / len(times)) * 1000
            total_ms = sum(times) * 1000
            count = len(times)
            percentage = (sum(times) / base_time * 100) if base_time > 0 else 0

            display_name = name.replace("Attn_", "")
            print(f"{display_name:<35} {avg_ms:>12.3f} {total_ms:>14.3f} {count:>8} {percentage:>6.1f}%")

        print("-"*80)
        avg_attn = (base_time / len(self.timings["Block_Attention"])) * 1000
        print(f"BASE: Block_Attention {'':<14} {avg_attn:>12.3f} {base_time*1000:>14.3f} {'':<8} {'100.0%':>7}")
        print("="*80 + "\n")

    def report_mlp_breakdown(self, title="MLP (Feed-Forward) Breakdown"):
        """Print detailed breakdown of MLP operations."""
        if not self.timings:
            print("No profiling data collected.")
            return

        mlp_components = [
            "MLP_Expansion",
            "MLP_Activation",
            "MLP_Projection"
        ]

        has_mlp_data = any(comp in self.timings for comp in mlp_components)
        if not has_mlp_data:
            return

        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80)

        if "Block_MLP" in self.timings:
            base_time = sum(self.timings["Block_MLP"])
            print(f"Percentages relative to: Block_MLP")
        else:
            print("Warning: Block_MLP not found")
            return

        print(f"{'Component':<35} {'Avg (ms)':>12} {'Total (ms)':>14} {'Count':>8} {'%':>7}")
        print("-"*80)

        mlp_timings = [(name, times) for name, times in self.timings.items()
                      if name in mlp_components]
        mlp_timings.sort(key=lambda x: sum(x[1]), reverse=True)

        for name, times in mlp_timings:
            avg_ms = (sum(times) / len(times)) * 1000
            total_ms = sum(times) * 1000
            count = len(times)
            percentage = (sum(times) / base_time * 100) if base_time > 0 else 0

            display_name = name.replace("MLP_", "")
            print(f"{display_name:<35} {avg_ms:>12.3f} {total_ms:>14.3f} {count:>8} {percentage:>6.1f}%")

        print("-"*80)
        avg_mlp = (base_time / len(self.timings["Block_MLP"])) * 1000
        print(f"BASE: Block_MLP {'':<20} {avg_mlp:>12.3f} {base_time*1000:>14.3f} {'':<8} {'100.0%':>7}")
        print("="*80 + "\n")

    def get_average(self, name):
        """Get average time for a specific component."""
        if name not in self.timings or not self.timings[name]:
            return 0.0
        return sum(self.timings[name]) / len(self.timings[name])

    def get_layer_breakdown(self):
        """Aggregate timing by layer number."""
        layer_times = defaultdict(list)

        for key, times in self.timings.items():
            if key.startswith("Layer_") and len(key.split("_")) >= 2:
                try:
                    layer_num = int(key.split("_")[1])
                    layer_times[layer_num].extend(times)
                except (ValueError, IndexError):
                    continue

        return layer_times


# Global profiler instance
profiler = GPT2Profiler()


# ============================================================================
#                     PROFILED ATTENTION FORWARD PASS
# ============================================================================

def profiled_attention_forward(attn_module, x):
    """
    Profiled forward pass for CausalMultiHeadSelfAttention.
    """
    B, T, C = x.size()

    # QKV Projection
    with profiler.profile("Attn_QKV_Projection"):
        k = attn_module.W_key(x)
        q = attn_module.W_query(x)
        v = attn_module.W_value(x)

    # Reshape for multi-head attention
    with profiler.profile("Attn_QKV_Reshape"):
        k = k.view(B, T, attn_module.n_head, C // attn_module.n_head).transpose(1, 2)
        q = q.view(B, T, attn_module.n_head, C // attn_module.n_head).transpose(1, 2)
        v = v.view(B, T, attn_module.n_head, C // attn_module.n_head).transpose(1, 2)

    # Compute attention scores
    with profiler.profile("Attn_Scores_Compute"):
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))

    # Apply causal mask
    with profiler.profile("Attn_Mask_Apply"):
        att = att.masked_fill(attn_module.bias[:, :, :T, :T] == 0, float('-inf'))

    # Softmax and dropout
    with profiler.profile("Attn_Softmax"):
        att = torch.nn.functional.softmax(att, dim=-1)
        att = attn_module.attn_dropout(att)

    # Apply attention to values
    with profiler.profile("Attn_Values_Apply"):
        y = att @ v

    # Reshape output
    with profiler.profile("Attn_Output_Reshape"):
        y = y.transpose(1, 2).contiguous().view(B, T, C)

    # Output projection
    with profiler.profile("Attn_Output_Projection"):
        y = attn_module.c_proj(y)

    return y


# ============================================================================
#                     PROFILED MLP FORWARD PASS
# ============================================================================

def profiled_mlp_forward(mlp_module, x):
    """
    Profiled forward pass for MLP (Feed-Forward Network).
    """
    # Expansion layer
    with profiler.profile("MLP_Expansion"):
        x = mlp_module.c_fc(x)

    # Activation function
    with profiler.profile("MLP_Activation"):
        x = mlp_module.act(x)

    # Projection layer
    with profiler.profile("MLP_Projection"):
        x = mlp_module.c_proj(x)

    return x


# ============================================================================
#                     PROFILED FORWARD PASS WRAPPER
# ============================================================================

def profiled_forward(model, idx):
    """
    Wrapper around model forward pass with detailed profiling.
    This doesn't modify your original model class.
    """
    with profiler.profile("Total_Forward_Pass"):
        device = idx.device
        b, t = idx.size()

        # Embeddings
        with profiler.profile("Embeddings"):
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
            tok_emb = model.wte(idx)
            pos_emb = model.wpe(pos)
            x = model.drop(tok_emb + pos_emb)

        # Transformer blocks
        with profiler.profile("All_Transformer_Blocks"):
            for layer_idx, block in enumerate(model.h):
                with profiler.profile(f"Layer_{layer_idx}_Total"):
                    # Attention path
                    shortcut = x

                    with profiler.profile("Block_LayerNorm_1"):
                        x = block.ln_1(x)

                    with profiler.profile("Block_Attention"):
                        x = profiled_attention_forward(block.attn, x)

                    x = block.drop_shortcut(x)
                    x = x + shortcut

                    # MLP path
                    shortcut = x

                    with profiler.profile("Block_LayerNorm_2"):
                        x = block.ln_2(x)

                    with profiler.profile("Block_MLP"):
                        x = profiled_mlp_forward(block.mlp, x)

                    x = block.drop_shortcut(x)
                    x = x + shortcut

        # Final layers
        with profiler.profile("Final_LayerNorm"):
            x = model.ln_f(x)

        with profiler.profile("LM_Head_Projection"):
            logits = model.lm_head(x)

    return logits


# ============================================================================
#                        MAIN PROFILING FUNCTION
# ============================================================================

def profile_gpt2_model(model, tokenizer, prompt="My name is Shubham Ojha",
                       num_warmup=5, num_runs=50, max_new_tokens=30):
    """
    Complete profiling of GPT-2 model.

    Args:
        model: Your GPT-2 model
        tokenizer: GPT-2 tokenizer
        prompt: Input text prompt
        num_warmup: Number of warmup iterations
        num_runs: Number of profiled iterations
        max_new_tokens: Number of tokens to generate
    """
    model.eval()
    device = next(model.parameters()).device

    print("\n" + "="*80)
    print("  GPT-2 124M MODEL PROFILING")
    print("="*80)
    print(f"Device: {device}")
    print(f"Prompt: '{prompt}'")
    print(f"Warmup runs: {num_warmup}")
    print(f"Profiled runs: {num_runs}")
    print(f"Tokens per run: {max_new_tokens}")
    print("="*80 + "\n")

    # Reset profiler
    profiler.reset()

    # ========== WARMUP ==========
    print("🔥 Warming up GPU...")
    with torch.no_grad():
        for _ in range(num_warmup):
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            for _ in range(min(10, max_new_tokens)):
                logits = model(input_ids)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)

    print("✓ Warmup complete\n")

    # ========== PROFILING ==========
    print(f"📊 Profiling {num_runs} inference runs...")

    all_tokens_generated = 0

    with torch.no_grad():
        for run_idx in range(num_runs):
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            tokens_this_run = 0

            for token_idx in range(max_new_tokens):
                logits = profiled_forward(model, input_ids)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                tokens_this_run += 1
                all_tokens_generated += 1

                if next_token.item() == tokenizer.eos_token_id:
                    break

            if (run_idx + 1) % 10 == 0:
                print(f"  ✓ Completed {run_idx + 1}/{num_runs} runs")

    print(f"✓ Profiling complete! Generated {all_tokens_generated} total tokens\n")

    # ========== REPORTS ==========

    # GPT-2 Level Report
    profiler.report(f"GPT-2 Level Profiling ({num_runs} runs, {all_tokens_generated} tokens)")

    # Block Level Report
    profiler.report_block_level(f"Block Level Profiling ({num_runs} runs, {all_tokens_generated} tokens)")

    # Attention Breakdown Report
    profiler.report_attention_breakdown(f"Attention Breakdown ({num_runs} runs, {all_tokens_generated} tokens)")

    # MLP Breakdown Report
    profiler.report_mlp_breakdown(f"MLP Breakdown ({num_runs} runs, {all_tokens_generated} tokens)")

    # Per-token statistics
    print("📈 PER-TOKEN STATISTICS")
    print("="*80)
    avg_forward = profiler.get_average("Total_Forward_Pass") * 1000
    print(f"Average forward pass:        {avg_forward:.3f} ms/token")

    if "All_Transformer_Blocks" in profiler.timings:
        avg_blocks = profiler.get_average("All_Transformer_Blocks") * 1000
        print(f"All transformer blocks:      {avg_blocks:.3f} ms/token ({avg_blocks/avg_forward*100:.1f}%)")

    tokens_per_sec = 1000 / avg_forward if avg_forward > 0 else 0
    print(f"\nThroughput:                  {tokens_per_sec:.2f} tokens/second")
    print("="*80 + "\n")

    print("✅ Profiling complete!\n")


# ============================================================================
#                              RUN PROFILING
# ============================================================================

if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    print("Loading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Run profiling
    profile_gpt2_model(
        model=model,
        tokenizer=tokenizer,
        prompt="What is your name",
        num_warmup=20,
        num_runs=20,
        max_new_tokens=30
    )