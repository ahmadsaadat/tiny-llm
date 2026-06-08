# Tiny-LLM

In this project we will learn how to create a large language model

# Intro
- An LLM is comprised of 2 parts:
    1. Training:
        - This part creates a model.
        - A model is a bunch of tables that have undergone many rounds of optimization in order to make them predict the next word more accurately when you pass in a sentence.
    2. Inference:
        - This is where you use that optimized model in order to make your sentence predictions.

## Training
1. Data: You have to gather lots of data and clean them
2. Tokenization: You have to represent your data as numbers/tokens (numbers are easier to do math on)
3. Tables: You then have to make a bunch of tables/ matrices that will learn patterns (these tables are like optimized galton board, remember normal distribution? They will guide the numbers to their destiny)
    - Embedding Table: These are optimized random numbers that represent token definition
    - Embedding Position Table: These are optimized random numbers that represent the token position (relative to other tokens in a sequence)
    - LM Head Table: These are optimized random numbers that make the next word prediction (based on the last vector in a sequence)
4. Transformer Blocks: You will pass the token embeddings through this block so that the model can build context
    - Layer Norm Table: Stabilizes the numbers.
    - Feed Forward Table: Helps transform and refine the token representation.
    - Attention Table: Helps token decide which previous tokens are important to remember.

