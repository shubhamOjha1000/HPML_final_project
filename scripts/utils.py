import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Load pretrained GPT-2 124M weights from HuggingFace into the custom model:- 

def load_pretrained_weights(model):
    """
    Load pretrained GPT-2 124M weights from HuggingFace into the custom model.

    Args:
        model: Your custom GPT2 model instance
    """
    from transformers import GPT2LMHeadModel

    print("Loading pretrained GPT-2 124M weights from HuggingFace...")

    # Load the pretrained HuggingFace model
    hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
    hf_state_dict = hf_model.state_dict()

    # Get your model's state dict
    custom_state_dict = model.state_dict()

    # Simple direct mappings (no transpose needed)
    simple_mappings = {
        # Token embeddings
        'transformer.wte.weight': 'wte.weight',

        # Position embeddings
        'transformer.wpe.weight': 'wpe.weight',

        # Final layer norm
        'transformer.ln_f.weight': 'ln_f.scale',
        'transformer.ln_f.bias': 'ln_f.shift',
    }

    # Load simple mappings
    for hf_name, custom_name in simple_mappings.items():
        if hf_name in hf_state_dict and custom_name in custom_state_dict:
            custom_state_dict[custom_name].copy_(hf_state_dict[hf_name])

    # Load each transformer block
    for i in range(model.config['n_layer']):
        # Layer norm 1
        custom_state_dict[f'h.{i}.ln_1.scale'].copy_(hf_state_dict[f'transformer.h.{i}.ln_1.weight'])
        custom_state_dict[f'h.{i}.ln_1.shift'].copy_(hf_state_dict[f'transformer.h.{i}.ln_1.bias'])

        # Attention QKV - split the combined c_attn
        c_attn_weight = hf_state_dict[f'transformer.h.{i}.attn.c_attn.weight']
        c_attn_bias = hf_state_dict[f'transformer.h.{i}.attn.c_attn.bias']

        n_embd = model.config['n_embd']

        # HuggingFace stores weights as (n_embd, 3*n_embd) - need to split along dim 1
        q_weight, k_weight, v_weight = c_attn_weight.split(n_embd, dim=1)
        q_bias, k_bias, v_bias = c_attn_bias.split(n_embd, dim=0)

        # Copy Q, K, V weights (transpose because HF format is transposed)
        custom_state_dict[f'h.{i}.attn.W_query.weight'].copy_(q_weight.T)
        custom_state_dict[f'h.{i}.attn.W_query.bias'].copy_(q_bias)

        custom_state_dict[f'h.{i}.attn.W_key.weight'].copy_(k_weight.T)
        custom_state_dict[f'h.{i}.attn.W_key.bias'].copy_(k_bias)

        custom_state_dict[f'h.{i}.attn.W_value.weight'].copy_(v_weight.T)
        custom_state_dict[f'h.{i}.attn.W_value.bias'].copy_(v_bias)

        # Attention output projection (transpose)
        custom_state_dict[f'h.{i}.attn.c_proj.weight'].copy_(
            hf_state_dict[f'transformer.h.{i}.attn.c_proj.weight'].T
        )
        custom_state_dict[f'h.{i}.attn.c_proj.bias'].copy_(
            hf_state_dict[f'transformer.h.{i}.attn.c_proj.bias']
        )

        # Layer norm 2
        custom_state_dict[f'h.{i}.ln_2.scale'].copy_(hf_state_dict[f'transformer.h.{i}.ln_2.weight'])
        custom_state_dict[f'h.{i}.ln_2.shift'].copy_(hf_state_dict[f'transformer.h.{i}.ln_2.bias'])

        # MLP - c_fc (transpose)
        custom_state_dict[f'h.{i}.mlp.c_fc.weight'].copy_(
            hf_state_dict[f'transformer.h.{i}.mlp.c_fc.weight'].T
        )
        custom_state_dict[f'h.{i}.mlp.c_fc.bias'].copy_(
            hf_state_dict[f'transformer.h.{i}.mlp.c_fc.bias']
        )

        # MLP - c_proj (transpose)
        custom_state_dict[f'h.{i}.mlp.c_proj.weight'].copy_(
            hf_state_dict[f'transformer.h.{i}.mlp.c_proj.weight'].T
        )
        custom_state_dict[f'h.{i}.mlp.c_proj.bias'].copy_(
            hf_state_dict[f'transformer.h.{i}.mlp.c_proj.bias']
        )

    print("✓ Pretrained weights loaded successfully!")

    # Verify weight sharing between wte and lm_head
    assert model.wte.weight.data_ptr() == model.lm_head.weight.data_ptr(), \
        "Weight sharing broken between wte and lm_head"

    return model












# Run Inference :-

def run_inference(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=50):
    """
    Run inference on a text prompt.

    Args:
        model: GPT-2 model
        tokenizer: Tokenizer
        prompt: Input text
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random, lower = more deterministic)
        top_k: Top-k sampling parameter (only sample from top k most likely tokens)

    Returns:
        Generated text
    """
    model.eval()  # Set model to evaluation mode
    device = next(model.parameters()).device  # Get model's device

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Store original length to know where generation starts
    original_length = input_ids.shape[1]

    print(f"Prompt: {prompt}")
    print(f"Generating {max_new_tokens} tokens...\n")

    # Generation loop
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Check if we've exceeded max context length
            if input_ids.shape[1] >= model.config['n_positions']:
                print("Warning: Reached maximum context length")
                break

            # Forward pass
            logits = model(input_ids)  # Shape: (batch_size, seq_len, vocab_size)

            # Get logits for the last token
            logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

            # Apply temperature scaling
            logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                # Get top-k logits and indices
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))

                # Create a mask for top-k tokens
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered.scatter_(1, top_k_indices, top_k_logits)
                logits = logits_filtered

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Optional: Stop if we generate an end-of-text token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the generated sequence
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    return generated_text

