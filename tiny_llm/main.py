import torch

from tiny_llm.modules.a_data import Data
from tiny_llm.modules.b_tables import Table
from tiny_llm.modules.c_trainer import Trainer
from tiny_llm.modules.d_transformer_block import TransformerBlock


def train():
    sequence_length = 16
    sequence_size_max = 20_000
    dimension_attention = 64
    dimension_embedding = 64
    number_of_transformer_blocks = 5

    # 1. load data, clean, tokenize and create training sequences
    data = Data(
        filename_input_data="tiny_llm/io/tiny_stories.txt",
        filename_output_data="tiny_llm/io/tiny_stories_tokenizer.json",
        sequence_length=sequence_length,
        sequence_size_max=sequence_size_max,
    )

    # 2. create your weight tables
    tables = [
        Table(
            vocab_size=data.tokenizer.get_vocab_size(),
            sequence_length=sequence_length,
            dimension_attention=dimension_attention,
            dimension_embedding=dimension_embedding,
        )
        for _ in range(number_of_transformer_blocks)
    ]

    # 3. train
    Trainer(
        tables=tables,
        dimension_attention=dimension_attention,
        training_sequences=data.training_sequences,
        sequence_length=sequence_length,
    ).train()


def infer(user_input: str):
    data = Data(
        filename_input_data="tiny_llm/io/tiny_stories.txt",
        filename_output_data="tiny_llm/io/tiny_stories_tokenizer.json",
        sequence_length=16,
        sequence_size_max=20_000,
    )

    checkpoint = torch.load(
        "tiny_llm/io/tiny_stories_gpt.pt",
        map_location="mps",
        weights_only=False,
    )

    tables = checkpoint["tables"]

    token_ids = data.tokenizer.encode(user_input.lower()).ids

    for _ in range(10):
        input_ids = token_ids[-16:]

        while len(input_ids) < 16:
            input_ids.insert(0, 0)

        inputs = torch.tensor([input_ids], device="mps")

        # embeddings + positions
        X = tables[0].table_embedding[inputs]
        Z = X + tables[0].table_position

        # transformer blocks
        for table in tables:
            Z = TransformerBlock(
                table,
                64,
                torch.device("mps"),
            ).forward(Z)

        # logits
        last_token_vector = Z[:, -1]
        logits = last_token_vector @ tables[0].table_lm_head

        # greedy pick
        next_token_id = torch.argmax(logits, dim=-1).item()

        token_ids.append(next_token_id)

    return data.tokenizer.decode(token_ids)


if __name__ == "__main__":
    user_input = input(
        "Press Enter to train, or type a message and press Enter to chat: "
    )

    if user_input.strip():
        while True:
            print(infer(user_input))
            user_input = input()
    else:
        train()
