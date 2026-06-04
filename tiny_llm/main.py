from tiny_llm.modules.a_data_prep import DataPrep
from tiny_llm.modules.b_weight_tables import create_weights
from tiny_llm.modules.c_training import training_loop
from tiny_llm.modules.utils import create_optimizer

if __name__ == "__main__":
    # 1. load data, clean, tokenize and create training sequences
    data = DataPrep(
        filename_input_data="tiny_llm/io/tiny_stories.txt",
        filename_output_data="tiny_llm/io/tiny_stories_tokenizer.json",
        sequence_length=16,
        sequence_size_max=20_000,
    )
    tokenizer = data.tokenizer
    sequences = data.training_sequences

    # 2. create your weight tables
    weights = create_weights(tokenizer, sequences)

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
