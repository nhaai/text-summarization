from nlp_utils import *

# ==============================
# OVERLAP SIMILARITY
# ==============================
def overlap_similarity(tokens_i, tokens_j):
    """
    Compute lexical overlap-based similarity between two sentences
    This is the classical TextRank similarity (non-vector-based)
    """
    # If either sentence is empty after tokenization, similarity is zero
    if not tokens_i or not tokens_j:
        return 0.0

    # Convert token lists to sets to remove duplicates
    set_i = set(tokens_i)
    set_j = set(tokens_j)

    # Intersection of tokens between two sentences
    overlap = set_i & set_j

    # If there is no common token, similarity is zero
    if not overlap:
        return 0.0

    # Overlap-based similarity formula:
    # sim(Si, Sj) = |Si ∩ Sj| / (log|Si| + log|Sj|)
    # This normalization prevents longer sentences from dominating
    return len(overlap) / (np.log(len(set_i)) + np.log(len(set_j)))

# ==============================
# OVERLAP-BASED TEXTRANK PIPELINE
# ==============================
def run_textrank(sentences, ratio=0.33, damping=0.85):
    """
    Steps:
    1. Tokenize sentences
    2. Compute overlap-based sentence similarity
    3. Build sentence graph using similarity matrix
    4. Apply PageRank to rank sentence importance
    5. Select top-K sentences for summary
    """
    # Stopwords are not explicitly removed in this overlap-based TextRank implementation.
    # This may increase lexical overlap between sentences and make its behavior closer
    # to TF-IDF-based PageRank for texts with homogeneous vocabulary.
    tokenized = [vi_tokenizer(s) for s in sentences]

    N = len(sentences)

    # Initialize similarity matrix
    sim_matrix = np.zeros((N, N))

    # Compute pairwise overlap similarity
    for i in range(N):
        for j in range(N):
            # Self-similarity is explicitly ignored (no self-loops)
            if i != j:
                sim_matrix[i][j] = overlap_similarity(
                    tokenized[i], tokenized[j]
                )

    # Build sentence similarity graph
    # - Nodes represent sentences
    # - Edge weights represent overlap similarity
    graph = nx.from_numpy_array(sim_matrix)

    # Apply PageRank on overlap-based sentence graph
    # PageRank captures sentence importance from lexical overlap structure
    scores = nx.pagerank(graph, alpha=damping)

    # Sort sentences by PageRank score (descending)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Select top-K sentences for summary
    top_k = max(1, int(len(sentences) * ratio))

    # Sentence indices selected by TextRank
    selected_ids = {i for i, _ in ranked[:top_k]}

    return {
        "tokens": tokenized,          # Tokenized sentences
        "sim_matrix": sim_matrix,     # Overlap-based similarity matrix
        "graph": graph,               # Sentence similarity graph
        "scores": scores,             # PageRank scores for each sentence
        "top_k": top_k,               # Number of selected sentences
        "selected_ids": selected_ids, # Selected sentence indices (TextRank)
    }
