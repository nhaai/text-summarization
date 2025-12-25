import re
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from underthesea import word_tokenize

# ==============================
# SENTENCE SPLITTING
# ==============================
S_TAG_PATTERN = re.compile(
    r'<s\s+docid="[^"]+"\s+num="\d+"\s+wdcount="\d+">.*?</s>',
    re.DOTALL
)

def split_sentences(text):
    if S_TAG_PATTERN.search(text):
        sentences = re.findall(r'<s[^>]*>(.*?)</s>', text, flags=re.DOTALL)
    else:
        text = text.replace("\r", "")
        lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 0]

        sentences = []
        for line in lines:
            parts = re.split(r'(?<=[.!?])\s+', line)
            sentences.extend(parts)

    return [s.strip() for s in sentences if len(s.split()) > 5]

# ==============================
# TOKENIZATION
# ==============================
def vi_tokenizer(sentence):
    tokens = word_tokenize(sentence, format="text").split()
    return [t.lower() for t in tokens if t.isalnum() or "_" in t]

# ==============================
# EXPORT GRAPH
# ==============================
def export_graph(graph, scores, png_path=None, gexf_path=None, title="Sentence Graph",with_labels=True):
    """
    Export sentence graph for visualization and analysis.
    - graph: networkx graph
    - scores: PageRank scores (dict)
    - png_path: path to save PNG (optional)
    - gexf_path: path to save GEXF for Gephi (optional)
    """
    for node in graph.nodes:
        graph.nodes[node]["pagerank"] = scores.get(node, 0.0)

    # export GEXF
    if gexf_path:
        nx.write_gexf(graph, gexf_path)

    # export PNG
    if png_path:
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(graph, seed=42)

        pr_values = np.array([scores.get(n, 0.0) for n in graph.nodes])
        if pr_values.max() > 0:
            pr_norm = pr_values / pr_values.max()
        else:
            pr_norm = pr_values

        nodes = nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=800,
            node_color=pr_norm,
            cmap=plt.get_cmap("YlOrRd")
        )

        nx.draw_networkx_edges(
            graph,
            pos,
            alpha=0.4
        )

        if with_labels:
            labels = {i: f"S{i+1}" for i in graph.nodes}
            nx.draw_networkx_labels(graph, pos, labels, font_size=9)

        plt.colorbar(nodes, label="PageRank Score")
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()

def save_graph(graph, path):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue')
    plt.savefig(path)
    plt.close()

def save_heatmap(sim_matrix, path):
    plt.figure(figsize=(6, 5))
    plt.imshow(sim_matrix, cmap='viridis')
    plt.colorbar()
    plt.title("Cosine Similarity Heatmap")
    plt.savefig(path)
    plt.close()
