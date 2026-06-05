from tiny_llm.modules.a_data import Data
from tiny_llm.modules.b_tables import Table
from tiny_llm.modules.c_trainer import Trainer

if __name__ == "__main__":
    sequence_length = 16
    sequence_size_max = 20_000
    dimension_attention = 64
    dimension_embedding = 64
    number_of_weight_tables = 2

    # 1. load data, clean, tokenize and create training sequences
    data = Data(
        filename_input_data="tiny_llm/io/tiny_stories.txt",
        filename_output_data="tiny_llm/io/tiny_stories_tokenizer.json",
        sequence_length=16,
        sequence_size_max=20_000,
    )

    # 2. create your weight tables
    tables = [
        Table(
            vocab_size=data.tokenizer.get_vocab_size(),
            sequence_length=16,
            dimension_attention=64,
            dimension_embedding=64,
        )
        for i in range(number_of_weight_tables)
    ]

    # 3. train
    trainer = Trainer(
        tables=tables,
        dimension_attention=64,
        training_sequences=data.training_sequences,
        sequence_length=16,
    ).train()
