from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def read_text(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def prepare_text(raw_text: str) -> list[str]:
    END_TOKEN = "<END>"

    lines = [
        line.replace("@-@", "-").replace("@,@", ",").replace("@.@", ".").strip().lower()
        for line in raw_text.splitlines()
        if line.strip()
    ]

    lines = [
        line for line in lines if not (line.startswith("=") and line.endswith("="))
    ]

    return [f"{line} {END_TOKEN}" for line in lines]


def tokenizer(text: list[str], unknown_character: str):
    tokenizer = Tokenizer(BPE(unk_token=unknown_character))

    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=20000,
        special_tokens=[
            "<UNK>",
            "<|endoftext|>",
        ],
    )

    tokenizer.train_from_iterator(
        prepared_text,
        trainer,
    )

    tokenizer.save("./output_model/tokenizer.json")

    return tokenizer


def training_sequences(
    prepared_text: list[str], tokenizer, sequence_length: int, max_sequences: int
) -> list[tuple]:
    training_sequences = []

    for line in prepared_text:
        token_ids = tokenizer.encode(line).ids

        if len(token_ids) <= sequence_length:
            continue

        for i in range(len(token_ids) - sequence_length):
            input_ids = token_ids[i : i + sequence_length]
            target_id = token_ids[i + sequence_length]

            training_sequences.append((input_ids, target_id))

            if len(training_sequences) >= max_sequences:
                break

        if len(training_sequences) >= max_sequences:
            break

        print("training_sequences: ", training_sequences[0])
        print("input: ", tokenizer.decode(training_sequences[0][0]))
        print("output: ", tokenizer.decode([training_sequences[0][1]]))

    return training_sequences
