import numpy as np

class Attention:
    """
    Single head self attention

    Formula
    -------

    scores = Q @ K.T
    weights = softmax(scores)
    output = weights @ V
    """

    def __init__(self, embedding_dimension: int, attention_dimension: int):
        """
        Create attention weight matrices

        get embedding dimension from embedding

        choose arbitrary number for attention dimensions


        W_K = Who am I ?
        W_Q = What am I looking for ?
        W_V = What do I pass forward ?
        """

        self.embedding_dimension = embedding_dimension
        self.attention_dimension = attention_dimension
                
        self.W_Q = np.random.randn(
            embedding_dimension,
            attention_dimension
        )

        self.W_K = np.random.randn(
            embedding_dimension,
            attention_dimension
        )

        self.W_V = np.random.randn(
            embedding_dimension,
            attention_dimension
        )
        
    def softmax(self, scores: np.ndarray):
        """
        convert scores into probabilities.
        """

        # Numerical stability trick
        scores = scores - np.max(
            scores,
            axis=-1,
            keepdims=True
        )

        exp_scores = np.exp(scores)

        probabilities = exp_scores / np.sum(
            exp_scores,
            axis=-1,
            keepdims=True
        )

        return probabilities
    
    def forward(self, embeddings: np.ndarray):
        """
        Run self-attention
        """

        # Create keys
        K = embeddings.dot(self.W_K)

        # Create Queries
        Q = embeddings.dot(self.W_Q)

        # Create Values
        V = embeddings.dot(self.W_V)

        # Compare every query against every key
        scores = Q.dot(K.T)

        # prevent scores from growing too large
        scores = scores/ np.sqrt(self.attention_dimension)

        # Convert scores into probabilities
        attention_weights = self.softmax(scores)

        # Mix value vectors together
        output = attention_weights @ V

        return output, attention_weights


if __name__ == "__main__":
    # Create fake embeddings

    embeddings = np.array([
        [0.2, 0.5, 0.1, 0.7],
        [0.9, 0.3, 0.8, 0.2],
        [0.4, 0.6, 0.2, 0.5]
    ])

    print("\nEMBEDDINGS:")
    print(embeddings)

    print("\nEMBEDDING SHAPE:")
    print(embeddings.shape)

    # Create attention layer

    attention = Attention(
        embedding_dimension=4,
        attention_dimension=2
    )

    # Run attention

    output, attention_weights = attention.forward(
        embeddings
    )

    # results

    print("\nATTENTION WEIGHTS:")
    print(attention_weights)

    print("\nATTENTION WEIGHT SHAPE:")
    print(attention_weights.shape)

    print("\nOUTPUT:")
    print(output)

    print("\nOUTPUT SHAPE:")
    print(output.shape)