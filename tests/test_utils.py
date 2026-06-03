import unittest

import torch
from tokenizers import Tokenizer

from tiny_llm.utils import (
    find_tokens,
    get_batch,
    get_batch_inputs,
    get_batch_targets,
    prepare_text,
    read_text,
    training_sequences,
)

FILENAME = "tiny_llm/input_data/mock_text.txt"
RAW_TEXT = """Spot saw the shiny car and said, Wow, Kitty, your car is so bright and clean!\nOnce upon a time, there was a kind farmer. He had a big cow. The cow was sad. The farmer did not know why.\nOnce upon a time, there was a little girl named Lucy. She had a pet cat named Tom."""
PREPARED_TEXT = [
    "spot saw the shiny car and said, wow, kitty, your car is so bright and clean! <END>",
    "once upon a time, there was a kind farmer. he had a big cow. the cow was sad. the farmer did not know why. <END>",
    "once upon a time, there was a little girl named lucy. she had a pet cat named tom. <END>",
]
UNKNOWN_CHARACTER = "<UNK>"
TOKENIZER_FILENAME = "tests/output_model/tokenizer.json"


class TestReadText(unittest.TestCase):
    def test_read_text(self):
        assert read_text(filename=FILENAME) == """hello 1\nhello 2\nhello 3"""


class TestPrepareText(unittest.TestCase):
    def test_prepare_text(self):
        assert prepare_text(raw_text=RAW_TEXT) == PREPARED_TEXT


class TestFindTokens(unittest.TestCase):
    def test_find_tokens(self):
        assert isinstance(
            find_tokens(
                text=PREPARED_TEXT,
                filename=TOKENIZER_FILENAME,
            ),
            Tokenizer,
        )


class TestTrainingSequences(unittest.TestCase):
    TEXT = [
        "Spot saw the shiny car and said, Wow, Kitty, your car is so bright and clean! <END>",
        "Once upon a time, there was a kind farmer. He had a big cow. The cow was sad. The farmer did not know why.",
        "Once upon a time, there was a little girl named Lucy. She had a pet cat named Tom.",
    ]
    FILENAME = "tests/output_model/tokenizer.json"
    TOKENIZER = find_tokens(text=TEXT, filename=FILENAME)

    SEQUENCE_LENGTH = 16
    MAX_SEQUENCE = 500_000

    def test_training_sequences(self):
        sequences = training_sequences(
            text=self.TEXT,
            tokenizer=self.TOKENIZER,
            sequence_length=self.SEQUENCE_LENGTH,
            max_sequences=self.MAX_SEQUENCE,
        )

        assert isinstance(sequences, list)
        assert isinstance(sequences[0], tuple)
        assert isinstance(sequences[0][0], list)
        assert isinstance(sequences[0][1], int)


# class TestTables
class TestTraining(unittest.TestCase):
    DEVICE = torch.device("mps")
    TEXT = [
        "Spot saw the shiny car and said, Wow, Kitty, your car is so bright and clean! <END>",
        "Once upon a time, there was a kind farmer. He had a big cow. The cow was sad. The farmer did not know why.",
        "Once upon a time, there was a little girl named Lucy. She had a pet cat named Tom.",
    ]
    FILENAME = "tests/output_model/tokenizer.json"
    TOKENIZER = find_tokens(text=TEXT, filename=FILENAME)
    SEQUENCE_LENGTH = 16
    MAX_SEQUENCES = 500_000
    TRAINING_SEQUENCES = training_sequences(
        text=TEXT,
        tokenizer=TOKENIZER,
        sequence_length=SEQUENCE_LENGTH,
        max_sequences=MAX_SEQUENCES,
    )

    def test_training(self):
        # batch of input ids and output
        batch = get_batch(self.TRAINING_SEQUENCES, step=1, batch_size=5)
        assert batch[0] == (
            [109, 46, 123, 52, 68, 110, 3, 82, 3, 112, 3, 124, 52, 95, 103, 116],
            68,
        )

        batch_input_ids = get_batch_inputs(batch=batch, device=self.DEVICE)
        assert torch.equal(
            batch_input_ids[0].cpu(),
            torch.tensor(
                [109, 46, 123, 52, 68, 110, 3, 82, 3, 112, 3, 124, 52, 95, 103, 116]
            ),
        )

        batch_target_ids = get_batch_targets(batch=batch, device=self.DEVICE)
        assert torch.equal(
            batch_target_ids[0].cpu(),
            torch.tensor(68),
        )
