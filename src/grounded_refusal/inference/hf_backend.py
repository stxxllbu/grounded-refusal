from __future__ import annotations


def run_sequential_inference(
    prompts: list[str],
    *,
    model_name: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    device_map: str = "auto",
) -> list[str]:
    """Run inference one prompt at a time (no batching -- fine at pilot scale)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )

    outputs: list[str] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature

        generated = model.generate(**inputs, **generate_kwargs)
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

    return outputs
