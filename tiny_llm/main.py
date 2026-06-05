from tiny_llm.modules.a_data import Data
from tiny_llm.modules.b_tables import Table
from tiny_llm.modules.c_trainer import Trainer


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
    pass


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
