from tiny_llm.modules.utils import (
    find_tokens,
    prepare_text,
    read_text,
    training_sequences,
)


def prepare_training_data(
    filename: str = "tiny_llm/io/tiny_stories.txt", sequence_length: int = 16
):
    # 1. Load text
    raw_text = read_text(filename)

    # 2 Prepare text
    text = prepare_text(raw_text=raw_text)

    # 3. Create Tokens
    tokenizer = find_tokens(text=text)

    # 4. build sequences
    sequences = training_sequences(
        text=text,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        max_sequences=500_000,
    )

    return tokenizer, sequences
