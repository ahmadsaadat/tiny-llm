from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

END_TOKEN = "<END>"
UNKNOWN_TOKEN = "<UNK>"


class Data:
    """
    All things data

    1. Loads data
    2. Cleans data
    3. Tokenizes data
    4. Creates training sequences
    """

    filename_input_data: str
    filename_output_data: str
    sequence_length: int
    sequence_size_max: int

    def __init__(
        self,
        filename_input_data: str,
        filename_output_data: str,
        sequence_length: int,
        sequence_size_max: int,
    ):
        self.filename_input_data = filename_input_data
        self.filename_output_data = filename_output_data
        self.sequence_length = sequence_length
        self.sequence_size_max = sequence_size_max

    @property
    def data_unprocessed(self):
        with open(self.filename_input_data, "r", encoding="utf-8") as f:
            return f.read()

    @property
    def data_processed(self):
        lines = []

        for line in self.data_unprocessed.splitlines():
            if line.strip():
                lines.append(line.strip().lower())

        return [f"{line} {END_TOKEN}" for line in lines]

    @property
    def tokenizer(self):
        tokenizer = Tokenizer(BPE(unk_token=UNKNOWN_TOKEN))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=20000,
            special_tokens=[UNKNOWN_TOKEN, END_TOKEN],
        )
        tokenizer.train_from_iterator(
            self.data_processed,
            trainer,
        )
        tokenizer.save(self.filename_output_data)
        return tokenizer

    @property
    def training_sequences(self) -> list[int]:
        training_sequences = []
        tokenizer = self.tokenizer

        for line in self.data_processed:
            token_ids = tokenizer.encode(line).ids

            if len(token_ids) <= self.sequence_length:
                continue

            for i in range(len(token_ids) - self.sequence_length):
                input_ids = token_ids[i : i + self.sequence_length]
                target_id = token_ids[i + self.sequence_length]

                training_sequences.append((input_ids, target_id))

                if len(training_sequences) >= self.sequence_size_max:
                    break

            if len(training_sequences) >= self.sequence_size_max:
                break

        return training_sequences
