from tiny_llm.modules.a_data import Data
from tiny_llm.modules.b_tables import Tables
from tiny_llm.modules.c_training import training_loop
from tiny_llm.modules.utils import create_optimizer

if __name__ == "__main__":
    # 1. load data, clean, tokenize and create training sequences
    data = Data(
        filename_input_data="tiny_llm/io/tiny_stories.txt",
        filename_output_data="tiny_llm/io/tiny_stories_tokenizer.json",
        sequence_length=16,
        sequence_size_max=20_000,
    )

    # 2. create your weight tables
    tables = Tables(
        size_vocab=data.tokenizer.get_vocab_size(),
        size_training_sequences=len(data.training_sequences),
        dimension_attention=64,
        dimension_embedding=64,
    )

    # 3. create your optimizer
    optimizer = create_optimizer(
        parameters=list(weights.values()),
        learning_rate=0.001,
    )

    # 4. training loop
    training_loop(
        sequences=sequences,
        weights=weights,
        optimizer=optimizer,
        sequence_length=16,
        attention_dimension=64,
        n_heads=4,
        epochs=6,
        batch_size=64,
    )
