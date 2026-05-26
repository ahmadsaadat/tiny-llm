class Tokenizer:
    """
    Converts text into token IDs
    """

    word_to_id: dict[str, int] = {}
    id_to_word: dict[int, str] = {}

    def __init__(self, training_sentences: list[str]):
        """
        Build dictionary
        """

        # Get list of words
        tokens = []
        for sentence in training_sentences:
            tokens.extend(sentence.lower().split())

        # Get unique words
        lexicon = sorted(set(tokens))

        # Word -> Integer
        self.word_to_id = {
            word: i for i, word in enumerate(lexicon)
        }

        # Integer -> Word
        self.id_to_word = {
            i: word for word, i in self.word_to_id.items()
        }


    def encode(self, sentence: str) -> list[int]:
        """
        Convert sentence -> token IDs
        """

        words = sentence.lower().split()

        ids = [
            self.word_to_id[word] for word in words
        ]

        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Convert token IDs -> text
        """

        words = [self.id_to_word[i] for i in ids]

        return " ".join(words)
    
    @property
    def vocab_size(self) -> int:
        return len(self.word_to_id)


if __name__ == "__main__":

    # Train
    tokenizer = Tokenizer(["hello world", "my Friend"])
    # Encode
    ids = tokenizer.encode("hello friend")
    assert ids == [1, 0]
    # Decode
    sentence = tokenizer.decode(ids)
    assert sentence == "hello friend"
