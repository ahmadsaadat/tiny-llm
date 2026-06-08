# Module 2: Predicting the Next Word
- Transforming embeddings into predictions

## What we'll learn
- The journey is straightforward: Embeddings -> Transformations -> Probabilities
![alt text](doc/image-16.png)
- Functions with Learnable parameters:
    - See how function like ``` output = weights @ input + bias ``` can learn patterns
- Matrix Multiplication:
    - Core operation that transforms vectors
    - An embedding [0.8, 0.2, 0.5] multiplied by a 768×50000 matrix, you get 50000 numbers—one score for each word in the vocabulary.
- From score to probabilities:
    - Connect matrix multiplication to language prediciton
- Softmax - Scores to Probabilities: Convert raw scores into probabilities that sum to 1.0
- Building a Next-Word predictor:
    - All these pieces combine.
    - You take a word embedding.
    - Transform it through matrix multiplication.
    - Apply softmax.
    - Output probabilities.
![alt text](doc/image-17.png)

# Functions with Learnable Parameters
## What is a Neural Network?
- A neural network is just a function
- A mathematical operation that takes inputs and produces outputs
- The special part? The function has parameters that can be adjusted to change its behavior.
![alt text](doc/image-18.png)
- A more befitting name is parameterized function 😆

## Simple example: Predicting House Prices
- f(n) -> price = size × weight + bias
- house_size = 1200
- weight = 150
- bias = 50000
- = 230,000
- Together, weight × input + bias creates a linear function. 
- This is the building block of all neural networks. 
- Complex models like GPT just use millions of these simple operations stacked together.

# Improving Prediction
- How do we find good values for weight and bias? Training.
- When predictions are wrong, the model adjusts the parameters slightly to reduce the error.
- This procedss is called gradient descent.

## The core building block:
- The examples above demonstrate the fundamental building block of neural networks: linear transformations using weights × inputs + bias.
- Small predictor based on Size, Bedroom, Age, Location, Bias: 5 parameters (4 weights + 1 bias)
- GPT-3: 175 billion parameters
- The next article explores the core operation that makes neural networks efficient: matrix multiplication.

# Matrix Multiplication - The Core Operation

## one input, multiple outputs
- imagine you have:
    - size of house
    - size of bedrooms
    - distance to city
    - age of house
- you want to find out:
    - predicted price
    - probability of selling within 30 days
    - expected days on market
- So with an input with vector shape of (1, 4) and weight matrix shape of (4, 3), one multiply can give you all 3 outputs at once:
- ![alt text](doc/image-19.png)
- if shapes do not align, check orientation and transpose the weight matrix

# Predicting the Next Word
## From House Prices to Words
- Remember the earlier example?
- 4 inputs (size, bedroom, location, age) predicted 3 outputs (price, probability of selling, days on market)
- Language prediction works the same way, instead of predicting 3 housing values, we predict 20 scores - one for each word in vocabulary.

## The Vocabulary Matrix
- Our vocabulary has 20 words:
    - the, cat, dog, sat, ran, on, mat, house, a, big, small, quickly, slowly, and, is, red, blue, to, PAD, END.
- We need 20 outputs - one score per word.
- The transformation matrix has 20 rows - one per vocabulary, and 64 columns - matching embedding dimension
- ![alt text](doc/image-20.png)

## What changes, what stays the same
- Never changes:
    - weight matrix (20 rows, 64 columns)
    - input embedding (64 rows, 1 column)
    - output (20 scores) 
- What changes between predictions is the input embedding, which 64 numbers we feed in - not the size.

# From Embeddings to Logits
- LM Head (Language Modeling Head): Basically imagine it's the 20 x 64 output weight matrix.
- Logits: The 20 raw scores produced by multiplying the LM Head by an embeddings. These are unnormalized numbers.

## Understanding Logits
- The logits are raw numbers with no particular range.
- e. The word with the highest logit is the most likely next word, but we can't directly interpret the magnitude.


# Softmax - Turning Scores into Probabilities
## From scores to probabilities
- After processing input through layers of matrix multiplications, an LLM produces raw scores for each word in its vocabulary.
- But scores alone aren't useful. You need probabilities.
- Softmax takes raw scores (any real numbers) and outputs a valid probability distribution.

## The Softmax Formula
- Softmax converts scores to probabilities in two steps:
    - Exponentiate each score: e^score
    - Normalize by dividing each by the sum of all exponentials
    - P(i) = e^sᵢ / (e^s₁ + e^s₂ + ... + e^sₙ)
- ![alt text](doc/image-21.png)

## Why exponentiation works
- Exponentiation amplifies differences between scores.
- Consider two scores: 1.0 and 2.0.
- The difference is small (1.0). But e^1.0 ≈ 2.7 and e^2.0 ≈ 7.4.
- The ratio is now ~2.7×, making the higher score stand out more.

## How LLMs Use Softmax
- Every time an LLM predicts the next token, softmax is the final step.
- The model computes scores for all vocabulary tokens.
- For a 100-word response, softmax runs 100+ times.

# Conclusion
- Words become vectors (embeddings)
- Matrix multiplication transforms those vectors.
- This predictor takes one word as input and predicts what word should come next.
- The architecture is remarkably simple: lookup the embedding for the input word, multiply it by a learned weight matrix, apply softmax to get probabilities, then pick the highest probability word.
- ![alt text](doc/image-22.png)

## Fundamental limitation
- Real language depends heavily on context. The word "bank" means different things in "river bank" versus "savings bank". 
- The predictor has no way to use surrounding words to disambiguate.
- Embeddings capture useful patterns. But this is far from what context-aware models achieve.
![alt text](doc/image-23.png)
- But the context-blindness limitation is fundamental.
- The model processes one word at a time, with no memory of what came before or anticipation of what comes next.
- This limitation motivates the attention mechanism we'll learn in Module 3. 
- Attention allows the model to look at multiple words simultaneously and use context to make better predictions.