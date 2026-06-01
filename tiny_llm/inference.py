import numpy as np
import torch
import torch.nn.functional as F

model_path = "tiny_llm/output_model/tiny_wiki_gpt.npz"

model = np.load(model_path, allow_pickle=True)

embedding_table = torch.tensor(model["embedding_table"], dtype=torch.float32)
W_Q = torch.tensor(model["W_Q"], dtype=torch.float32)
W_K = torch.tensor(model["W_K"], dtype=torch.float32)
W_V = torch.tensor(model["W_V"], dtype=torch.float32)
lm_head = torch.tensor(model["lm_head"], dtype=torch.float32)

vocab = model["vocab"].tolist()
sequence_length = int(model["sequence_length"])
attention_dimension = int(model["attention_dimension"])

word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}


def predict_next(text, max_new_words=30):
    words = text.lower().split()

    for _ in range(max_new_words):
        input_ids = [word_to_id[word] for word in words if word in word_to_id]
        input_ids = input_ids[-sequence_length:]

        if len(input_ids) == 0:
            print("No known words found in input.")
            return text

        input_ids = torch.tensor(input_ids)

        with torch.no_grad():
            X = embedding_table[input_ids]

            Q = X @ W_Q
            K = X @ W_K
            V = X @ W_V

            scores = Q @ K.T
            scores = scores / torch.sqrt(torch.tensor(attention_dimension))

            mask = torch.triu(torch.ones(scores.shape), diagonal=1)
            scores = scores.masked_fill(mask == 1, float("-inf"))

            attention_weights = F.softmax(scores, dim=-1)
            context = attention_weights @ V

            last_token_vector = context[-1]
            logits = last_token_vector @ lm_head

            probabilities = F.softmax(logits, dim=-1)

            # sampling instead of greedy argmax
            logits = logits / 0.8
            probabilities = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1).item()
            next_word = id_to_word[next_id]

        words.append(next_word)

    return " ".join(words)


print(predict_next("the united states"))
print(predict_next("the history of"))
print(predict_next("the city of"))
