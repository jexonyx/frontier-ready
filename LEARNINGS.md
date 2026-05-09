# Learnings

Notes on concepts I've worked through and now understand. Written in my own words to consolidate understanding.

## Attention Mechanism (Q/K/V)

Each token's embedding is projected through three learned weight matrices to produce Query, Key, and Value vectors.

- **Query (Q):** What information is this token looking for from other tokens?
- **Key (K):** What does this token advertise about itself for relevance matching?
- **Value (V):** What semantic content does this token actually pass along when attended to?

The Q-K dot product determines *how much* to attend (affinity score). A high score amplifies the contribution of that token's Value to the output. The output for each token is a weighted sum of all Value vectors it can see, where the weights come from softmax-normalized Q-K scores.

K and V are separate because *why a token is relevant* (K) is different from *what information you want to extract from it* (V). A verb's query might match a noun's key due to syntactic subject-verb affinity, but the value vector passed along encodes different features — animacy, number, or whatever helps predict the next token.

**Multi-head attention** runs multiple sets of Q/K/V projections in parallel. Each head learns different relationship types (syntactic, positional, coreference, etc.), and their outputs are concatenated and projected back down. This lets the model capture multiple types of relationships simultaneously.

**Caveat:** The model doesn't literally learn human-interpretable concepts. Q/K/V matrices are optimized end-to-end by gradient descent to minimize next-token prediction loss. Sometimes the learned representations align with linguistic concepts we can name, sometimes they don't.
