# 🚦 Text Summarization

Pipeline A - TF-IDF
- Step 1: Sentence segmentation
- Step 2: TF-IDF vectorization
- Step 3: Cosine similarity computation
- Step 4: Sentence similarity graph construction
- Step 5: PageRank sentence ranking (graph-based)
- Step 6: Summary generation using PageRank
- Step 7: Logistic Regression scoring using TF-IDF
- Step 8: Comparative analysis between PageRank-based and Logistic-Regression-based summaries

Pipeline B - TextRank
- Step 1: Sentence segmentation
- Step 2: Tokenization (no explicit stopword removal)
- Step 3: Overlap-based sentence similarity computation
- Step 4: Sentence similarity graph construction
- Step 5: PageRank sentence ranking
- Step 6: Summary generation using TextRank

---

## 📌 1. System Requirements

### Ubuntu 24.04 LTS on WSL
Install required system libraries:

```bash
sudo apt install -y libnss3 libasound2t64
```

### Python, Node.js
Tested on Python v3.9.2

### Python Dependencies
```bash
python3.9 -m venv .venv39
source .venv39/bin/activate
pip install -r requirements.txt
```

---

## 📂 2. Project Structure

```bash
project/
│
├── summarizer.py            # Core logic (8 steps)
│
├── app.py                   # Flask + Tailwind demo UI
├── static/
│   └── uploads/
├── templates/
│   └── index.html
|
├── requirements.txt         # Python dependencies (pip)
└── README.md
```
---

## 🌐 3. Demo

Run:

```bash
python3 app.py
```

Open browser:

```
http://127.0.0.1:5000/
```

---

## 📝 Notes

| No. | Requirement                                                                                       |  Score   |
|----:|---------------------------------------------------------------------------------------------------|:--------:|
|   1 | Clearly define the problem objective and specify the input/output                                 |   1.0    |
|   2 | Describe the approach used to solve the problem and explain the main idea                         |   1.0    |
|   3 | Describe in detail the steps of the chosen approach                                               |   1.0    |
|   4 | Implement at least 5 features for data representation / represent text as a graph                 |   2.0    |
|   5 | Apply a machine learning classification method **or** rank nodes in the graph based on importance |   2.0    |
|   6 | Successfully generate a text summary                                                              |   1.0    |
|   7 | Analyze and evaluate the results: accuracy, strengths, and weaknesses                             |   1.0    |
|   8 | Propose improvements to the method, such as adding features or adjusting graph weights            |   1.0    |
|     | **Total**                                                                                         | **10.0** |
