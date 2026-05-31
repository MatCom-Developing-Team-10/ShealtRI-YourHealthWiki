"""Information-retrieval evaluation metrics.

Pure functions implementing the objective evaluation measures taught in the
course lecture (Conf_4 — "Evaluación, retroalimentación y expansión de consulta").
Every function is side-effect free and operates on plain Python types so it can
be unit-tested in isolation.

Lecture notation (confusion matrix for an IR system, where the system acts as a
binary classifier "relevant / not relevant" for a given query):

    +-----------------+-------------+---------------+
    |                 |  Relevant   |  Irrelevant   |
    +-----------------+-------------+---------------+
    | Retrieved       |     RR      |      RI       |
    | Not retrieved   |     NR      |      NI       |
    +-----------------+-------------+---------------+

    RR = REL ∩ REC   (relevant retrieved)
    RI = REC \\ RR    (retrieved but irrelevant)
    NR = REL \\ RR    (relevant but not retrieved)
    NI = NN          (irrelevant not retrieved)

From these, the lecture defines:

    Precision  P  = |RR| / |RR ∪ RI|     (fraction of retrieved that is relevant)
    Recall     R  = |RR| / |RR ∪ NR|     (fraction of relevant that was retrieved)
    F-measure  F  = (1 + β²)·P·R / (β²·P + R)
    F1            = 2·P·R / (P + R)        (special case β = 1)

Ranking-aware measures (which P/R/F1 ignore because they disregard order):

    DCG@k   = Σ_{i=1..k}  rel_i / log₂(i + 1)
    NDCG@k  = DCG@k / IDCG@k
    AP(q)   = (1/|R_q|) · Σ_{k=1..n} P@k · rel(k)
    MRR     = (1/|Q|) · Σ_i 1 / rank_i      (aggregated over queries; see service.py)
"""

from __future__ import annotations

import math
from collections import namedtuple

# Relevance judgments are a mapping doc_id -> graded relevance (>= 1 means
# relevant; 0 or absent means not relevant). Binary judgments are the special
# case where every relevant document has grade 1.
RelevanceMap = dict[str, int]

# Container for the four confusion-matrix regions described in the lecture.
# NI is optional because it requires knowing the full corpus size; most
# ranking metrics do not need it.
ConfusionCounts = namedtuple("ConfusionCounts", ["rr", "ri", "nr", "ni"])


def _relevant_ids(relevance: RelevanceMap) -> set[str]:
    """Return the set of doc_ids judged relevant (graded relevance >= 1)."""
    return {doc_id for doc_id, grade in relevance.items() if grade >= 1}


def confusion_counts(
    retrieved: list[str],
    relevance: RelevanceMap,
    corpus_size: int | None = None,
) -> ConfusionCounts:
    """Compute the RR / RI / NR / NI regions of the IR confusion matrix.

    Args:
        retrieved: Ordered list of retrieved doc_ids (order is ignored here;
            duplicates are collapsed).
        relevance: Relevance judgments (qrels) for the query.
        corpus_size: Total number of documents in the collection. Required to
            compute NI (irrelevant not retrieved); if None, ``ni`` is set to 0.

    Returns:
        A :class:`ConfusionCounts` namedtuple ``(rr, ri, nr, ni)`` with the
        cardinalities |RR|, |RI|, |NR|, |NI|.
    """
    retrieved_set = set(retrieved)
    relevant = _relevant_ids(relevance)

    rr = len(retrieved_set & relevant)        # RR = REL ∩ REC
    ri = len(retrieved_set - relevant)         # RI = REC \ RR
    nr = len(relevant - retrieved_set)         # NR = REL \ RR

    if corpus_size is None:
        ni = 0
    else:
        # NI = corpus minus everything else (relevant retrieved + retrieved
        # irrelevant + relevant not retrieved).
        ni = max(0, corpus_size - (rr + ri + nr))

    return ConfusionCounts(rr=rr, ri=ri, nr=nr, ni=ni)


def precision(retrieved: list[str], relevance: RelevanceMap) -> float:
    """Precision P = |RR| / |RR ∪ RI|.

    Fraction of retrieved documents that are relevant.

    Returns:
        Precision in [0, 1]. Returns 0.0 when nothing is retrieved (the lecture
        notes precision is *undefined* in that case; we map it to 0.0 so the
        metric stays aggregable).
    """
    counts = confusion_counts(retrieved, relevance)
    denominator = counts.rr + counts.ri  # |RR ∪ RI| = all retrieved
    if denominator == 0:
        return 0.0
    return counts.rr / denominator


def recall(retrieved: list[str], relevance: RelevanceMap) -> float:
    """Recall R = |RR| / |RR ∪ NR|.

    Fraction of relevant documents that were retrieved.

    Returns:
        Recall in [0, 1]. Returns 0.0 when there is no relevant document (the
        lecture notes recall is *undefined* in that case).
    """
    counts = confusion_counts(retrieved, relevance)
    denominator = counts.rr + counts.nr  # |RR ∪ NR| = all relevant
    if denominator == 0:
        return 0.0
    return counts.rr / denominator


def fallout(
    retrieved: list[str],
    relevance: RelevanceMap,
    corpus_size: int,
) -> float:
    """Fallout = |RI| / |RI ∪ NI|.

    Fraction of the irrelevant documents that the system retrieved (the
    IR analogue of the false-positive rate). Lower is better; 0.0 means
    no irrelevant document was returned.

    Args:
        retrieved: Retrieved doc_ids (order ignored).
        relevance: Relevance judgments for the query.
        corpus_size: Total number of documents in the collection. Required
            to know how many irrelevant documents exist outside the
            retrieved set.

    Returns:
        Fallout in [0, 1]. Returns 0.0 when the collection contains no
        irrelevant documents.
    """
    counts = confusion_counts(retrieved, relevance, corpus_size=corpus_size)
    denominator = counts.ri + counts.ni  # all irrelevant docs in the collection
    if denominator == 0:
        return 0.0
    return counts.ri / denominator


def fallout_at_k(
    retrieved: list[str],
    relevance: RelevanceMap,
    corpus_size: int,
    k: int,
) -> float:
    """Fallout considering only the top-k of the ranking (Fallout@k)."""
    if k <= 0:
        return 0.0
    return fallout(retrieved[:k], relevance, corpus_size)


def f_measure(p: float, r: float, beta: float = 1.0) -> float:
    """General F-measure F = (1 + β²)·P·R / (β²·P + R).

    Args:
        p: Precision.
        r: Recall.
        beta: Weighting factor. ``beta == 1`` weights P and R equally (F1);
            ``beta > 1`` favours recall; ``beta < 1`` favours precision.

    Returns:
        F-measure in [0, 1]. Returns 0.0 when both P and R are 0.
    """
    beta_sq = beta * beta
    denominator = beta_sq * p + r
    if denominator == 0:
        return 0.0
    return (1 + beta_sq) * p * r / denominator


def f1(p: float, r: float) -> float:
    """F1 = 2·P·R / (P + R) — the F-measure with β = 1."""
    return f_measure(p, r, beta=1.0)


def precision_at_k(retrieved: list[str], relevance: RelevanceMap, k: int) -> float:
    """Precision considering only the top-k of the ranking (P@k).

    Args:
        retrieved: Ranked list of doc_ids (most relevant first).
        relevance: Relevance judgments.
        k: Cut-off rank.

    Returns:
        Fraction of the top-k that is relevant, in [0, 1].
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    relevant = _relevant_ids(relevance)
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], relevance: RelevanceMap, k: int) -> float:
    """Recall considering only the top-k of the ranking (R@k)."""
    if k <= 0:
        return 0.0
    relevant = _relevant_ids(relevance)
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevance: RelevanceMap) -> float:
    """Reciprocal rank 1 / rank for a single query.

    The reciprocal of the position of the *first* relevant document. The mean
    over a set of queries gives MRR (computed in the service layer).

    Returns:
        1 / rank of the first relevant hit, or 0.0 if none is relevant.
    """
    relevant = _relevant_ids(relevance)
    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def average_precision(retrieved: list[str], relevance: RelevanceMap) -> float:
    """Average Precision AP(q) = (1/|R_q|) · Σ_k P@k · rel(k).

    Precision is sampled at each rank where a relevant document appears, then
    averaged over the number of relevant documents. The mean of AP over all
    queries is MAP (Mean Average Precision), computed in the service layer.

    Args:
        retrieved: Ranked list of doc_ids.
        relevance: Relevance judgments.

    Returns:
        Average precision in [0, 1]. Returns 0.0 if there are no relevant docs.
    """
    relevant = _relevant_ids(relevance)
    if not relevant:
        return 0.0

    hits = 0
    score_sum = 0.0
    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            score_sum += hits / index  # P@k at this relevant position
    return score_sum / len(relevant)


def dcg_at_k(retrieved: list[str], relevance: RelevanceMap, k: int) -> float:
    """Discounted Cumulative Gain DCG@k = Σ_{i=1..k} rel_i / log₂(i + 1).

    Uses graded relevance (rel_i) so highly relevant documents contribute more
    than marginally relevant ones, and earlier positions are discounted less.

    Args:
        retrieved: Ranked list of doc_ids.
        relevance: Graded relevance judgments (grade used directly as rel_i).
        k: Cut-off rank.

    Returns:
        DCG@k (unbounded above; 0.0 if nothing relevant in the top-k).
    """
    dcg = 0.0
    for index, doc_id in enumerate(retrieved[:k], start=1):
        grade = relevance.get(doc_id, 0)
        if grade > 0:
            dcg += grade / math.log2(index + 1)
    return dcg


def ndcg_at_k(retrieved: list[str], relevance: RelevanceMap, k: int) -> float:
    """Normalized DCG@k = DCG@k / IDCG@k.

    IDCG@k is the DCG of the ideal ranking (relevance grades sorted descending),
    so NDCG@k == 1.0 means the system ordered the top-k exactly by relevance.

    Args:
        retrieved: Ranked list of doc_ids produced by the system.
        relevance: Graded relevance judgments.
        k: Cut-off rank.

    Returns:
        NDCG@k in [0, 1]. Returns 0.0 if there is no relevant document.
    """
    dcg = dcg_at_k(retrieved, relevance, k)

    # Ideal DCG: rank documents by descending relevance grade.
    ideal_grades = sorted(
        (grade for grade in relevance.values() if grade > 0), reverse=True
    )
    idcg = 0.0
    for index, grade in enumerate(ideal_grades[:k], start=1):
        idcg += grade / math.log2(index + 1)

    if idcg == 0:
        return 0.0
    return dcg / idcg
