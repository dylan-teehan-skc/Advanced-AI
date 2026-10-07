"""Scoring code for every run: cross-entropy (primary metric) and balanced accuracy (secondary).

Every function takes logits (N x K scores straight from the head, before softmax) and labels (N class indices),
as numpy arrays or torch tensors, and returns plain Python floats. Cross-entropy is in nats (natural log),
so guessing uniformly over 10 classes scores ln(10) = 2.303.

Self-check: python -m src.metrics
"""
import numpy as np


def _arrays(logits, labels):
    """Accept torch tensors or numpy arrays; return float64 logits (N x K) and int labels (N)."""
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    logits, labels = np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    assert logits.ndim == 2 and labels.shape == (len(logits),), (logits.shape, labels.shape)
    return logits, labels


def _nll(logits, labels):
    """Per-image negative log-likelihood of the true class: -log softmax(logits)[true class]."""
    z = logits - logits.max(axis=1, keepdims=True)                  # shift so exp() cannot overflow
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))    # log-softmax
    return -log_probs[np.arange(len(labels)), labels]


def cross_entropy(logits, labels):
    """Primary metric: mean negative log-likelihood of the true class over all images, in nats."""
    logits, labels = _arrays(logits, labels)
    return float(_nll(logits, labels).mean())


def per_class_cross_entropy(logits, labels):
    """Mean cross-entropy within each class. Array of length K, nan for a class with no images."""
    logits, labels = _arrays(logits, labels)
    nll = _nll(logits, labels)
    return np.array([nll[labels == c].mean() if (labels == c).any() else np.nan for c in range(logits.shape[1])])


def class_averaged_cross_entropy(logits, labels):
    """Fallback primary metric if the test set is uneven: average within each class, then across classes."""
    per_class = per_class_cross_entropy(logits, labels)
    assert not np.isnan(per_class).any(), "a class has no images, so the class average would skip it"
    return float(per_class.mean())


def balanced_accuracy(logits, labels):
    """Secondary metric: accuracy within each class (recall), averaged across classes."""
    logits, labels = _arrays(logits, labels)
    correct = logits.argmax(axis=1) == labels
    assert len(np.unique(labels)) == logits.shape[1], "a class has no images, so the class average would skip it"
    return float(np.mean([correct[labels == c].mean() for c in range(logits.shape[1])]))


if __name__ == "__main__":
    from sklearn.metrics import balanced_accuracy_score, log_loss

    close = lambda a, b: abs(a - b) < 1e-9

    # 1. uniform guess over 10 classes = ln(10) nats, whatever the label
    assert close(cross_entropy(np.zeros((10, 10)), np.arange(10)), np.log(10))

    # 2. hand-computed: true class 0 given probability 0.20, 0.05, 0.005 (the lecture 6 table)
    probs = np.array([[0.20, 0.60, 0.20], [0.05, 0.80, 0.15], [0.005, 0.900, 0.095]])
    labels = np.array([0, 0, 0])
    assert np.allclose(_nll(np.log(probs), labels), [1.6094379, 2.9957323, 5.2983174])
    assert close(cross_entropy(np.log(probs), labels), (1.6094379124 + 2.9957322736 + 5.2983173665) / 3)

    # 3. confident and correct -> essentially 0; confident and wrong -> large
    sure = np.array([[50.0, 0.0], [0.0, 50.0]])
    assert cross_entropy(sure, [0, 1]) < 1e-12 and cross_entropy(sure, [1, 0]) > 49

    # 4. uneven classes: 3 images of class 0 (true-class prob 0.5), 1 image of class 1 (true-class prob 0.25)
    logits = np.log(np.array([[0.5, 0.5]] * 3 + [[0.75, 0.25]]))
    labels = np.array([0, 0, 0, 1])
    ln2, ln4 = np.log(2), np.log(4)
    assert close(cross_entropy(logits, labels), (3 * ln2 + ln4) / 4)             # every image counts equally
    assert close(class_averaged_cross_entropy(logits, labels), (ln2 + ln4) / 2)  # every class counts equally
    assert np.allclose(per_class_cross_entropy(logits, labels), [ln2, ln4])

    # 5. balanced accuracy differs from plain accuracy when classes are uneven: 9/9 right and 0/1 right
    logits = np.array([[1.0, 0.0]] * 10)                    # always predicts class 0
    labels = np.array([0] * 9 + [1])
    assert close((logits.argmax(1) == labels).mean(), 0.9)  # plain accuracy
    assert close(balanced_accuracy(logits, labels), 0.5)    # (9/9 + 0/1) / 2

    # 6. random logits: agree with scikit-learn's independent implementations
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(500, 10)) * 3
    labels = rng.integers(0, 10, 500)
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    assert abs(cross_entropy(logits, labels) - log_loss(labels, probs, labels=list(range(10)))) < 1e-9
    assert close(balanced_accuracy(logits, labels), balanced_accuracy_score(labels, logits.argmax(1)))

    # 7. float16 logits (as saved features produce) and a missing class are both handled
    assert abs(cross_entropy(logits.astype(np.float16), labels) - cross_entropy(logits, labels)) < 1e-2
    assert np.isnan(per_class_cross_entropy(np.zeros((2, 3)), [0, 1])[2])

    # 8. same number as the loss PyTorch trains on, and torch tensors are accepted
    import torch

    t_logits, t_labels = torch.tensor(logits), torch.tensor(labels)
    assert abs(cross_entropy(t_logits, t_labels) - torch.nn.functional.cross_entropy(t_logits, t_labels).item()) < 1e-9
    assert close(balanced_accuracy(t_logits, t_labels), balanced_accuracy(logits, labels))

    print("metrics OK")
