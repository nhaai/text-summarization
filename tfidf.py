import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from nlp_utils import *

# ==============================
# LOAD REMOTE STOPWORDS
# ==============================
def load_stopwords(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    words = []
    for line in response.text.splitlines():
        line = line.strip().lower()
        if not line or line.startswith("#"):
            continue
        words.extend(line.split())
    return list(set(words))

STOPWORDS_EN_URL = "https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt"
STOPWORDS_VI_URL = "https://raw.githubusercontent.com/stopwords-iso/stopwords-vi/master/stopwords-vi.txt"
STOPWORDS_ALL = load_stopwords(STOPWORDS_EN_URL) + load_stopwords(STOPWORDS_VI_URL)

# ==============================
# LOGISTIC REGRESSION (PSEUDO-LABEL)
# ==============================
def run_logistic_classifier(tfidf, pseudo_labels):
    """
    Train a Logistic Regression classifier using pseudo-labels.
    Input:
    - tfidf        : TF-IDF sentence matrix
    - pseudo_labels: set of sentence indices labeled as important
    Output:
    - probs: probability of each sentence being important
    """
    # Feature matrix, each row corresponds to one sentence
    X = tfidf.toarray()

    # Pseudo-label assignment: label = 1 → important sentence, label = 0 → non-important sentence
    y = np.array([
        1 if i in pseudo_labels else 0
        for i in range(len(X))
    ])

    # Train Logistic Regression classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    # Predict probability of importance for each sentence, probs[i] ∈ (0, 1)
    probs = clf.predict_proba(X)[:, 1]

    return probs

# ==============================
# TF-IDF + PAGERANK PIPELINE
# ==============================
def run_tfidf(sentences, ratio=0.33, damping=0.85):
    """
    Steps:
    1. Convert sentences into TF-IDF vectors
    2. Compute cosine similarity between sentence vectors
    3. Build a sentence similarity graph
    4. Apply PageRank to estimate sentence importance
    5. Select top-K sentences as summary candidates
    """
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        tokenizer=vi_tokenizer,
        ngram_range=(1, 2), # unigrams + bigrams | (1, 1) → only unigrams, (2, 2) → only bigrams
        stop_words=STOPWORDS_ALL
    )
    tfidf = vectorizer.fit_transform(sentences)

    # Cosine similarity matrix (TF-IDF is L2-normalized, so dot product = cosine similarity)
    sim_matrix = cosine_similarity(tfidf)

    # Remove self-similarity (self-loops)
    np.fill_diagonal(sim_matrix, 0)

    # Build sentence similarity graph
    # - Nodes: sentences
    # - Edges: cosine similarity weights
    graph = nx.from_numpy_array(sim_matrix)

    # Apply PageRank on the sentence graph
    # PageRank captures sentence importance from global graph structure
    scores = nx.pagerank(graph, alpha=damping)

    # Sort sentences by PageRank score (descending)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Select top-K sentences for summary
    top_k = max(1, int(len(sentences) * ratio))

    # Sentence indices selected by PageRank
    selected_ids = {i for i, _ in ranked[:top_k]}

    return {
        "tfidf": tfidf,                # TF-IDF sentence matrix
        "sim_matrix": sim_matrix,      # Cosine similarity matrix
        "graph": graph,                # Sentence similarity graph
        "scores": scores,              # PageRank scores for each sentence
        "top_k": top_k,                # Number of selected sentences
        "selected_ids": selected_ids,  # Selected sentence indices (PageRank)
        "vectorizer": vectorizer,      # Fitted TF-IDF vectorizer
        "feature_names": vectorizer.get_feature_names_out(), # Vocabulary terms used in TF-IDF
        "idf_values": vectorizer.idf_, # IDF values corresponding to feature_names
    }
