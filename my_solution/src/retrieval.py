import statistics
import numpy as np
from . import config
from .data_classes import retrieved_chunk
from . import embedding


def retrieve_chunks(query, chunks, top_k=config.TOP_K):
    """Return the `top_k` chunks most similar to the query, each with its similarity score."""

    query_embedding = embedding.embed_texts([query], is_query=True)[0]
    # Cosine similarity of the query with every chunk (dot product, as vectors have length 1)
    similarity = np.array([query_embedding @ chunk.embedding for chunk in chunks])
    # Indices of the `top_k` most similar chunks, sorted by similarity
    top_indices = similarity.argsort()[-top_k:][::-1]
    # Build a new record per chunk with its score (the vector is no longer needed); the originals stay unchanged
    return [
        retrieved_chunk(**chunks[i].model_dump(exclude={"embedding"}), score=float(similarity[i]))
        for i in top_indices
    ]


def identify_retrieval_thresholds(questions, chunks, top_k=3, thresholds=np.arange(0.10, 0.90, 0.05)):
    """Try each evidence threshold on `questions` and return the one that best separates answerable from unanswerable questions."""

    # Retrieve once per question; each threshold then only filters these results
    retrieved = {q.id: retrieve_chunks(q.question, chunks, top_k=top_k) for q in questions}
    answerable = [q for q in questions if not q.should_abstain]
    unanswerable = [q for q in questions if q.should_abstain]

    results = []
    for threshold in thresholds:
        threshold = round(float(threshold), 2)
        # Evidence = the retrieved chunks that score at or above the threshold
        evidence = {qid: [c for c in found if c.score >= threshold] for qid, found in retrieved.items()}
        # Decision rule: answer if at least one chunk passes the threshold, otherwise abstain
        correct_decisions = sum(bool(evidence[q.id]) != q.should_abstain for q in questions)
        results.append({
            "threshold": threshold,
            "abstain_rate": statistics.mean(not evidence[q.id] for q in unanswerable),
            "decision_accuracy": correct_decisions / len(questions),
        })
    # Several thresholds can tie for the best accuracy: pick the middle one,
    # so the threshold sits as far as possible from both groups of questions
    best_accuracy = max(row["decision_accuracy"] for row in results)
    best_thresholds = [row["threshold"] for row in results if row["decision_accuracy"] == best_accuracy]
    chosen_threshold = best_thresholds[len(best_thresholds) // 2]

    return chosen_threshold