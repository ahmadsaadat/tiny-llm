
# Module 1 - From Text to Numbers - How LLMs convert text into numbers.

## Training vs Inference
- Training: The process of learning from billions of examples is called training. This happens once, before the model is deployed.
- Inference: When you use ChatGPT or Claude, you're using the model in inference mode. The model isn't learning anymore, It's applying the patterns it already learned to predict your next word.

## What LLMs Can't do
- They don't have real-time information - The model's knowledge freezes at its training cutoff date. It can't access current news, stock prices, or your private data unless you explicitly provide it.
- They make mistakes confidently - LLMs hallucinate - they generate plausible-sounding but incorrect information. They don't distinguish between reliable and unreliable text.
- They don't truly understand - The model predicts text based on statistical patterns. It doesn't understand meaning the way humans do.
- They're not deterministic - The same prompt can produce different outputs. Randomness is built into the sampling process.

## LLM's core architecture: the transformer
- The transformer is a specific design - a pipeline of mathematical operations that converts text into predictions.
- This is the blueprint that defines how ChatGPT, Claude, and GPT-4 process text. 
- Transformer Pipeline:
- ![alt text](doc/image.png)
- The pipeline runs every time you ask the model to predict the next word.

- Tokenization: Breaks text into tokens.
- Embeddings: Converts tokens into vectors.
- Transformer blocks: 2 layers of attention and feed-forward networks.
    - Attention: finds relationships between words
    - Feed-Forward Networks: Transform each word's representation based on its context.
- Prediction layer: "Given everything I learned about this context, how likely is the next word?"

## The Architecture is Universal
- The transformer architecture also powers:
    - Vision Models
    - Code generation

## Transformer Pipeline

![alt text](doc/image-1.png)

# Vectors & Arrays [Lists]
- What is a vector? vector is just a list of numbers. Nothing fancy. e.g. [2.5, -1.0, 3.7]
- Numbers in a vector can be positive, negative or zero. The only requirement is that a vector contains numbers in a specific order.
- The count of numbers in a vector is called the dimension/length e.g. above has a dimension of 3.
- Real transformer models user much longer vectors - typically 768 or 1024 numbers per word.

## How Transformers Use Vectors
- Transformers represent words as high-dimensional vectors. Instead of treating "cat" as a string of characters, the model uses a 768-dimensional vector like:

![alt text](doc/image-2.png)

- Vectors capture meaning through numerical patterns. Related words have similar vectors, e.g. cat and dog similar vectors, but cat and car would differ

## Vector Operations: Addition, Scaling and Dot Product

- Transformers perform three basic operations on vectors billions of times: addition, scaling and dot product.
- These 3 simple operations combine to create the complex behavior you see in ChatGPT

## Vector Addition: Combine Information
- Add two vectors by adding their corresponding elements. The vectors must have the same dimension.
- You cannot add vectors with different dimensions.
![alt text](doc/image-3.png)

## Scalar Multiplication: Scale Magnitude
- Every element gets multiplied by a scalar
![alt text](doc/image-4.png)

## Dot Product: Measure Similarity
- Multiplies elements by corresponding elements
![alt text](doc/image-5.png)

# Vectors/Lists turned into Matrices/Tables
- Vectors are lists of numbers - 1D arrays
- Matrices are table of numbers - 2D arrays

## What is a matrix
- A matrix is a rectangular grid of numbers arranged in rows and columns
![alt text](doc/image-6.png)

## Understanding Shape
- Shape is the most improtant property of a matrix. It tells you the dimensions and helps you understand what transformations are possible.
- Embedding Matrix: vocabulary -> word vectors e.g. stores one vector per word.
- Weight Matrix: transform 3D vectors -> 2D vectors e.g. transforms vectors from one dimension to another.
- Identity Matrix: leaves vectors unchanged -> same number of rows and columns.
- Real transformers use much larger embedding matrices (50257, 768)

## Vectors vs Matrices 
- A vector with shape (3,) is different from matrix with shape (1, 3) e.g. (1, 2, 3) vs ([1, 2, 3])
- Transformers sometimes need to reshape vectors into matrices for certain operations.
- The reshape function changes shape without changing the underlying data.

# Matrix Operations: Adding, Scaling and Multiplying

## Matrix Addition: Combine Information
- ![alt text](doc/image-7.png)

## Scalar Multiplication: Scale Every Element
- ![alt text](doc/image-8.png)

## Matrix Multiplication: Transform Data
- Multiply all the elements in a row on the left side with one element of column on the right side.
![alt text](doc/image-9.png)

## Compatibility Rules:
- Not all matrices can be multiplied together. The shapes must be compatible.
- For A @ B to work, the number of rows on left = columns on right
![alt text](doc/image-10.png)
- If A has shape (m, n) and B has shape (n, p), the result has shape (m, p)
- Shape errors are the most common mistake when working with matrices.
- Always check shapes before multiplying, use print(matrix.shape) frequently when debugging.

## Matrix-Vector Multiplication:
- ![alt text](doc/image-11.png)
- This is how transformers apply transformation to individual vectors.
- A weight matrix transforms an input vector into an output vector.

## Identity Matrix:
- The identity matrix is a special matrix that leaves other matrices unchanges when you multiply by it.
- It's analogous to multiplying a number by 1.
![alt text](doc/image-12.png)

## Combining Matrix Operations
- Operations have precendence: Matrix multiplications happens before addition.
- Use parentheses when needed to control order.
- (A + B) @ C differs from A + (B @ C)

# Conclusion
- Vectors: list of numbers []
- Vector operations: addition, scaling, dot product

- Matrices: table of numbers ([], [])
- Matrices operations: addition, scaling, multiplication

# Path Ahead

- Text -> Tokens/Subword Tokens -> Embeddings -> Similarity

- Text: input
- Tokenization: words turn into tokens
- Embeddings: tokens becomes vectors
- vector similarity: measure relationships between different vectors

## Text
- Tokenization is the process of splitting a sentence into words
- Sometimes a token is a word, part of a word or a punctuation
- ![alt text](doc/image-13.png)

## The Vocabulary Problem
- Both character-level and word-level tokenization have trade-offs:
- Character-level:
    - Small vocabulary (~100 tokens)
    - Can represent any text
    - Very inefficient - many tokens per sentence
    - Model must learn spelling patterns
- Word-level:
    - Large vocabulary (100,000+ tokens)
    - Efficient - fewer tokens per sentence
    - Cannot handle unknown words
    - Struggles with misspellings and new terms

- e.g. if "unbreakable" isn't in your vocabulary, world-level tokenization fails.
- Real LLMs use a middle ground approach called subword tokenization.

## Tokens vs Words
- Tokens are not always words. Depending on the tokenization method:
    - Word can be multiple tokens: "unbreakable" -> ["un", "break", "able"]
    - Multiple words might be one token: "New York" -> ["NewYork"]
    - Punctuation is usually separate tokens: "Hello!" -> ["Hello", "!"]
    - Numbers can be split "12345" -> ["12", "345"]
- When you use an LLM API, you're often charged per token.
- Understanding tokenization helps you estimate costs.
- A 100-word prompt might be 130 tokens or 80 tokens depending on the complexity of your vocabulary.

## Why this matters
- Tokenization affects everything about how LLMs work
- Performance: fewer tokens mean faster processing
- Context length: models have token limits, if your tokenization is inefficient your user can fit less content
- Cost: API pricing is per token
- Model Behavior: The model only sees tokens, not raw text. If "ChatGPT" is one token but "ClaudeAI" is three tokens, the model treats them differently.
- Rare words: Words not in the vocabulary become multiple tokens or unknown markers. This affects how well the model handles specialized domains.

# Subword Tokenization (BPE)
- Previously we saw two extremes: character-level tokenization and word-level tokenization
- Modern LLMs use a middle approach called subword tokenization: Byte-Pair Encoding (BPE)

## What is Byte-Pair Encoding?
- Repeatedly merge the most frequent pair of characters e.g. "t" + "h" -> "th"

Initial tokens:
  ['l', 'o', 'w']
  ['l', 'o', 'w', 'e', 's', 't']
  ['l', 'o', 'w', 'e', 'r']

Merge 1: 'l' + 'o' (appears 3 times)
Result:
  ['lo', 'w']
  ['lo', 'w', 'e', 's', 't']
  ['lo', 'w', 'e', 'r']

Merge 2: 'lo' + 'w' (appears 3 times)
Result:
  ['low']
  ['low', 'e', 's', 't']
  ['low', 'e', 'r']

Merge 3: 'low' + 'e' (appears 2 times)
Result:
  ['low']
  ['lowe', 's', 't']
  ['lowe', 'r']

## Building the Vocabulary:
- After running BPE, you have a vocabulary of tokens. 
- Think of this as the model's dictionary.

## Encoding New Text:
- Once you have the vocabulary from training, encoding new text is straightforward.
- Just repeat what you did earlier.

## Variations and Modern Approaches:
- Modern LLMs use variants of BPE:
    - WordPiece
    - SentencePiece
    - Byte-level BPE

# Embeddings - Words as Vectors
- An embedding is a vector/list that represents a token.
- The word "Ahmad" might map to the point (0.2, -0.5, 0.8) in 3 dimensional space.
- Similar words end up close together in this high-dimensional space
- example embeddings:
    - "cat" → [0.7, 0.2, -0.1, 0.5]
    - "dog" → [0.6, 0.3, -0.2, 0.4]
    - "table" → [-0.3, 0.8, 0.5, -0.2]
    - "chair" → [-0.2, 0.7, 0.6, -0.3]
- ![alt text](doc/image-14.png)

## Measuring Similarity
- How do we measure if two embeddings are similar? We calculate their distance
- Euclidean distance/ straight-line distance
- Small distance means high similarity. 
- Another common measure is cosine similarity, looks at angle between vectors vs absolute distance.

## Embedding Dimensions
- Real LLMs use large embeddings:
    - GPT-2 small: 768 dimensions
    - GPT-2 large: 1,280 dimensions
    - GPT-3: 12,288 dimensions (for the largest model)

# Vector Similarity
## Measuring Similarity
- LLMs typically use a measure called cosine similarity
- Cosine similarity looks at the angle between vectors rather than absolute distance
- Why? because if you did magnitude two vectors might miss each other, even though they could just be scaled by 3, e.g. [1, 2] vs [3, 6]
- ![alt text](doc/image-15.png)

## What is cosine similarity
- cosine_similarity = (A · B) / (||A|| × ||B||)
    - 1.0: Identical direction (very similar)
    - 0.0: Perpendicular (unrelated)
    - -1.0: Opposite direction (very dissimilar)

## Practical Applications:
- Cosine similarity powers:
    - Semantic Search
    - Recommendation Systems
    - Duplicate Detection
    - Clustering

## What we learnt
- Sentence - (tokenization)-> Tokens
- Tokens - (embedding) -> Embeddings/List/Vector
- Embeddings - (cosine similarity) -> similarity