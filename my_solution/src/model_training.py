import torch
from . import config
from .data_classes import instruction_case
from .generation import build_prompt


def build_training_text(tokenizer, case: instruction_case) -> str:
    """Join the answer-time prompt and the expected answer of one training case into a single text."""

    # Exactly the prompt the model sees when answering, filled with this case's question and context
    prompt = build_prompt(tokenizer, case.question, case.context)
    # The end-of-turn token (<|im_end|>) after the answer teaches the model to stop once it has answered
    return prompt + case.answer + tokenizer.eos_token


def tokenize_training_case(tokenizer, case: instruction_case, max_length=config.MAX_SEQUENCE_LENGTH):
    """Turn one training case into input_ids, attention_mask and labels, each of shape (1, number of tokens).

    The labels are a copy of input_ids with -100 on every prompt token, so the model is graded on the answer only.
    """

    # Full training text (prompt + answer + <|im_end|>) -> token IDs; the chat template already contains the special tokens
    encoded = tokenizer(build_training_text(tokenizer, case), return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    # The prompt alone, to find the position where the answer starts
    prompt_ids = tokenizer(
        build_prompt(tokenizer, case.question, case.context), return_tensors="pt", add_special_tokens=False
    )["input_ids"]
    prompt_length = prompt_ids.shape[1]

    # The boundary is only correct if the full text starts with exactly the prompt tokens
    if not torch.equal(input_ids[:, :prompt_length], prompt_ids):
        raise ValueError(f"training case {case.id}: the training text does not start with the prompt tokens")
    # Cutting the text short would remove (part of) the answer, the only graded part, so stop instead
    if input_ids.shape[1] > max_length:
        raise ValueError(f"training case {case.id} has {input_ids.shape[1]} tokens, more than max_length={max_length}")

    # Same token IDs, but -100 on the prompt positions: the loss ignores those
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100
    return input_ids, attention_mask, labels
