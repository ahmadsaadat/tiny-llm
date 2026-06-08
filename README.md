# Tiny-LLM

In this project we will learn how to create a large language model

## Contents
- [Intro](#intro)
- [Training](#training)
- [Inference](#inference)
- [Installation and Running](#installation-and-running)
- [Q/A](#qa)

## Intro
- An LLM is comprised of 2 parts:
    1. Training:
        - This part creates a model.
        - A model is a bunch of tables that have undergone many rounds of optimization in order to make them predict the next word more accurately when you pass in a sentence.
    2. Inference:
        - This is where you use that optimized model in order to make your sentence predictions.

## Training
1. Data: You have to gather lots of data and clean them
2. Tokenization: You have to represent your data as numbers ( aka tokens, cuz they're easier to do math on)
3. Tables: You then have to make a bunch of tables/ matrices that will learn patterns (these tables are like an optimized galton board, remember normal distribution? They will guide the numbers to their destiny)
    - Embedding Table: These are optimized random numbers that represent a token's meaning.
    - Embedding Position Table: These are optimized random numbers that represent the token's position (relative to other tokens in a sequence).
    - LM Head Table: These are optimized random numbers that make the next word prediction.
4. Transformer Blocks: You will pass the token embeddings through this block so that the model can build context
    - Layer Norm Table (beta/gamma): Stabilizes the numbers, such that they are between mean=0, std=1.
    ```
    layer_norm = gamma * ((x - mean) / (std + eps)) + beta
    ```
    - Attention Table (Q, K, V, O): Helps decide which tokens to pay more attention to.
    - Feed Forward Table (W1, W2, b1, b2): Helps decide what to do with those numbers.
    - Residual Connection: Helps retain certain numbers.
5. Loss: Measures how wrong the prediction was versus the actual sequence.
6. Optimization: Adjusts all table values so that next prediction is better.
7. Repeat: Keep repeating with different sequence until the avg loss gets closer to 0.
    ```
    avg_loss = total_loss / num_batches
    ```
8. Output: Output a model with all the optimized tables.

## Inference:
1. Load: Get the model and tokenizer outputted from training
2. Tokenization: Tokenize the user input
3. Tables: Convert the tokens into embeddings using the model
4. Transformer Block: Pass the embeddings through the training transformer block
5. Repeat: Keep predicting until the model outputs ```<END>```

## Installation and Running:
    1. ```poetry install```
    2. ```poetry run python tiny_llm/main.py```

## Q/A:
- How does one know what is the best params to create a really good model?
    - It is a combination of:
        1. Trial and Error (aka hyperparameter tuning)
        2. Research papers
    - What are some common hyperparameter I could tweak?
        1. Number of transformer blocks
        2. Embedding Dimension
        3. Attention Dimension
        4. Number of LM heads
        5. Learning Rate
        6. Batch Size
        7. Sequence Length
- As an professional model trainer, what should I look for?
    - Average Loss
    - Inference Quality
    - Training speed
    - Memory Usage