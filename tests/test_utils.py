import unittest

from tokenizers import Tokenizer

from tiny_llm.utils import find_tokens, prepare_text, read_text, training_sequences


class TestReadText(unittest.TestCase):
    FILENAME = "tiny_llm/input_data/mock_text.txt"
    EXPECTED = """hello 1\nhello 2\nhello 3"""

    def test_read_text(self):
        assert read_text(filename=self.FILENAME) == self.EXPECTED


class TestPrepareText(unittest.TestCase):
    RAW_TEXT = """hello 1\nhello 2\nhello 3"""
    EXPECTED = ["hello 1 <END>", "hello 2 <END>", "hello 3 <END>"]

    def test_prepare_text(self):
        assert prepare_text(raw_text=self.RAW_TEXT) == self.EXPECTED


class TestFindTokens(unittest.TestCase):
    TEXT = ["hello 1 <END>", "hello 2 <END>", "hello 3 <END>"]
    UNKNOWN_CHARACTER = "<UNK>"
    FILENAME = "tests/output_model/tokenizer.json"

    EXPECTED = Tokenizer

    def test_find_tokens(self):
        assert isinstance(
            find_tokens(
                text=self.TEXT,
                filename=self.FILENAME,
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

    TOKENS = find_tokens(text=TEXT, filename=FILENAME)

    SEQUENCE_LENGTH = 16
    MAX_SEQUENCE = 500_000

    def test_training_sequences(self):
        sequences = training_sequences(
            text=self.TEXT,
            tokenizer=self.TOKENS,
            sequence_length=self.SEQUENCE_LENGTH,
            max_sequences=self.MAX_SEQUENCE,
        )

        assert isinstance(sequences, list)
        assert isinstance(sequences[0], tuple)
        assert isinstance(sequences[0][0], list)
        assert isinstance(sequences[0][1], int)
