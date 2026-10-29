import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from . import config
from .data_classes import retrieved_chunk


def load_model(adapter_path=None):
    """Load the base model and its tokenizer; if `adapter_path` is given, add the LoRA adapter on top."""

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME, revision=config.BASE_MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        revision=config.BASE_MODEL_REVISION,
        # Same reasoning as the embedding model: CPU and float32 give the same output on every run
        device_map="cpu",
        dtype=torch.float32,
    )
    if adapter_path is not None:
        # Wraps the base model; the base weights stay unchanged, the small adapter weights are added
        model = PeftModel.from_pretrained(model, str(adapter_path))
    # Inference mode: switches off training-only behaviour such as dropout
    model.eval()
    return model, tokenizer


def format_context(chunks: list[retrieved_chunk], threshold: float) -> str:
    """Turn retrieved chunks into one passage per line, each starting with its source ID, e.g. '[refunds] ...'."""
    # Also remove chunks below the similarity threshold
    chunks = [chunk for chunk in chunks if chunk.score >= threshold]
    # Same layout as the `context` field of the training examples, so the adapter sees the same format
    return "\n".join(f"[{chunk.document_name}] {chunk.text}" for chunk in chunks)


def build_prompt(tokenizer, question: str, context: str) -> str:
    """Build the full prompt text in the model's chat format. Shared by generation and (later) LoRA training."""

    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": f"<information>\n{context}\n</information>\n\nQuestion: {question}"},
    ]
    # The chat template adds the model's special role markers; add_generation_prompt=True ends
    # the text with the opening of an assistant turn, so the model continues with the answer
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_answer(question: str, context: str, model, tokenizer, max_new_tokens=config.MAX_NEW_TOKENS) -> str:
    """Answer `question` from `context` with the given model (base or base + LoRA adapter)."""

    prompt = build_prompt(tokenizer, question, context)
    # Text -> token IDs; the chat template already contains the special tokens, so do not add them twice
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    # No gradients needed when only generating: faster and uses less memory
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # Greedy decoding: always pick the most likely next token, so the same input gives the same answer
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # The output repeats the prompt tokens first; keep only the newly generated answer tokens
    answer_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
