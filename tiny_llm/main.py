import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from tiny_llm.modules.a_data import Data
from tiny_llm.modules.b_tables import Table
from tiny_llm.modules.c_trainer import Trainer
from tiny_llm.modules.d_transformer_block import TransformerBlock

sequence_length = 16
sequence_size_max = 200_000
dimension_attention = 128
dimension_embedding = 128
number_of_transformer_blocks = 5
device = torch.device("mps")

# inference
temperature = 0.8
max_safety_tokens = 200  # only prevents infinite loop


def train():
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
    # load trained tables
    training_state = torch.load("tiny_llm/io/tiny_stories_gpt.pt", weights_only=False)
    training_tables = training_state["tables"]

    # turn input into token IDs
    tokenizer = Tokenizer.from_file("tiny_llm/io/tiny_stories_tokenizer.json")
    input_token_ids = tokenizer.encode(user_input.lower()).ids
    output_token_ids = []

    for i in range(max_safety_tokens):
        # if user input < 16  pad it with zeros
        input_preceding_token_ids = input_token_ids[-16:]
        input_preceding_token_ids = [0] * (
            16 - len(input_preceding_token_ids)
        ) + input_preceding_token_ids
        input_preceding_token_ids = torch.tensor(
            [input_preceding_token_ids], device=device
        )

        # converts tokens into embeddings
        X = training_tables[0].table_embedding[input_preceding_token_ids]
        Z = X + training_tables[0].table_position
        # run transformer blocks
        for table in training_tables:
            Z = TransformerBlock(table, dimension_attention, device).forward(Z)

        # predict next token
        last_token_vector = Z[:, -1]
        logits = last_token_vector @ training_tables[0].table_lm_head

        # pick next token
        probs = F.softmax(logits / temperature, dim=-1)
        next_id = torch.multinomial(probs[0], 1).item()

        next_token = tokenizer.id_to_token(next_id)

        # stop or continue
        if next_token == "<END>":
            break

        input_token_ids.append(next_id)
        output_token_ids.append(next_id)

    return tokenizer.decode(output_token_ids)


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
