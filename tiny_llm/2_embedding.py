import numpy as np

class Embedding:
    """
    Converts token IDs into vectors
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Create embedding table.

        vocab_size: number of unique words
        embedding_dim: number of features per word
        """
        
        # create random embedding matrix
        self.embedding_table = np.random.randn(vocab_size, embedding_dim)

    def forward(self, token_ids: list[int]):
        """
        Convert token IDs to have embeddings.
        """

        embedding = self.embedding_table[token_ids]

        return embedding

if __name__ == "__main__" :


    # Create random embedding table [prior to training]
    embedding_table = Embedding(vocab_size=4, embedding_dim=32)

    # Convert token ids to have embeddings
    embeddings = embedding_table.forward([1, 0])

    print("\nEMBEDDINGS:")
    print(embedding_table)

    print("\nSHAPE:")
    print(embeddings)