import random

import torch
import torch.nn.functional as F
from torch.optim.adam import Adam
from tqdm import tqdm

from tiny_llm.modules.b_tables import Table
from tiny_llm.modules.d_transformer_block import TransformerBlock


class Trainer:
    optimizer: Adam  # Adaptive Moment Estimation

    def __init__(
        self,
        dimension_attention: int,
        tables: list[Table],
        training_sequences: list,
        sequence_length: int,
        learning_rate: float = 0.001,
        device=torch.device("mps"),
    ):
        self.dimension_attention = dimension_attention
        self.tables = tables
        self.training_sequences = training_sequences
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.device = device

        all_parameters = []
        for table in self.tables:
            all_parameters.extend(table.tables_dict.values())

        self.optimizer = torch.optim.Adam(
            params=all_parameters,
            lr=learning_rate,
        )

    def train(
        self,
        training_rounds: int = 10,
        batch_size: int = 64,
    ):
        for round in range(training_rounds):
            total_loss = 0
            random.shuffle(self.training_sequences)
            for index in tqdm(range(0, len(self.training_sequences), batch_size)):
                # get batch of inputs and targets
                batch = self.training_sequences[index : index + batch_size]
                inputs = torch.tensor([x[0] for x in batch], device=self.device)
                target = torch.tensor([x[1] for x in batch], device=self.device)

                # forward pass the batch of inputs and targets
                ## get Z = embeddings + positions
                input_tables = self.tables[0]
                X = input_tables.table_embedding[inputs]
                Z = X + input_tables.table_position

                for table in self.tables:
                    ## transformer block
                    Z = TransformerBlock(
                        table,
                        self.dimension_attention,
                        self.device,
                    ).forward(Z)

                last_token_vector = Z[:, -1]
                logits = last_token_vector @ input_tables.table_lm_head
                loss = F.cross_entropy(logits, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            torch.save({"tables": self.tables}, "tiny_llm/io/tiny_gpt.pt")
            num_batches = len(range(0, len(self.training_sequences), batch_size))
            print("avg_loss:", total_loss / num_batches)
