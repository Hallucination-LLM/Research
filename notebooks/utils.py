import torch

def get_response(model, inputs, max_new_tokens=10, return_attentions=True):
    attentions = []

    for _ in range(max_new_tokens):
        outputs = model.forward(**inputs, output_attentions=True)

        attentions.append(outputs.attentions)

        next_token_logits = outputs.logits[:, -1, :]
        next_token_ids = next_token_logits.argmax(dim=-1).unsqueeze(-1)

        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token_ids], dim=-1)

        new_attention_mask = torch.ones_like(next_token_ids, device=inputs["attention_mask"].device)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], new_attention_mask], dim=-1)
    
    if return_attentions:
        return inputs["input_ids"], attentions
    return inputs["input_ids"]

def calc_lookback_ratio(attentions):
    n_layers = len(attentions[0])
    n_heads = attentions[0][0].shape[1]
    generated_len = len(attentions)

    lookback_ratio = torch.zeros((n_layers, n_heads, generated_len))

    prompt_len = attentions[0][0].shape[-1]
    
    for i in range(generated_len):
        for l in range(n_layers):
            attn_on_context = attentions[i][l][0, :, -1, :prompt_len].mean(-1)
            attn_on_new_tokens = attentions[i][l][0, :, -1, prompt_len:].mean(-1)
            lookback_ratio[l, :, i] = attn_on_context / (attn_on_context + attn_on_new_tokens)
            
    return lookback_ratio

def calc_lookback_ratio(attentions):
    
    n_layers = len(attentions[0])
    n_heads = attentions[0][0].shape[1]
    generated_len = len(attentions)
    prompt_len = attentions[0][0].shape[-1] - 1  # Initial sequence length minus 1

    # Initialize the result tensor
    lookback_ratio = torch.zeros((n_layers, n_heads, generated_len))

    for i in range(generated_len):
        # Stack attentions for all layers at this generation step
        layer_attentions = torch.stack([attentions[i][l] for l in range(n_layers)])
        
        # Calculate attention on context and new tokens
        attn_on_context = layer_attentions[:, 0, :, -1, :prompt_len].mean(-1)
        attn_on_new_tokens = layer_attentions[:, 0, :, -1, prompt_len:].mean(-1)
        
        # Calculate lookback ratio for this generation step
        lookback_ratio[:, :, i] = attn_on_context / (attn_on_context + attn_on_new_tokens)

        # Update prompt_len for the next iteration
        prompt_len += 1

    return lookback_ratio