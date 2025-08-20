import pytest
import torch

from entitynet.tasks.contrastive_task import compute_retrieval_cosine


def test_compute_retrieval_cosine_all_correct():
    """Test with identity matrix - all retrievals should be correct"""
    n = 5
    # Identity matrix: diagonal elements are highest for each row
    dot = torch.eye(n, dtype=torch.float32)

    metrics, other = compute_retrieval_cosine(dot)

    # All retrievals should be correct (rank 0)
    assert metrics["r1"] == 1.0  # 100% recall@1
    assert metrics["r5"] == 1.0  # 100% recall@5
    assert metrics["medr"] == 1.0  # median rank should be 1
    assert metrics["meanr"] == 1.0  # mean rank should be 1
    assert metrics["n"] == n

    # All ranks should be 0 (perfect retrieval)
    assert torch.all(other["ranks"] == 0)
    # Top1 predictions should be the correct indices
    assert torch.all(other["top1"] == torch.arange(n))


def test_compute_retrieval_cosine_all_wrong():
    """Test with off-diagonal matrix - all retrievals should be wrong"""
    n = 5
    # Create matrix where diagonal is lowest, off-diagonal is highest
    dot = torch.ones(n, n, dtype=torch.float32)
    dot.fill_diagonal_(0.0)  # Diagonal elements are lowest

    metrics, other = compute_retrieval_cosine(dot)

    # All retrievals should be wrong (rank > 0)
    assert metrics["r1"] == 0.0  # 0% recall@1
    # For n=5, r5 should be 1.0 since all ranks are 4 and r5 counts ranks < 5
    assert metrics["r5"] == 1.0  # 100% recall@5 (all ranks are 4, which is < 5)
    assert metrics["medr"] == n  # median rank should be n
    assert metrics["meanr"] == n  # mean rank should be n
    assert metrics["n"] == n

    # All ranks should be n-1 (worst possible rank)
    assert torch.all(other["ranks"] == n - 1)
    # Top1 predictions should not be the correct indices
    assert not torch.any(other["top1"] == torch.arange(n))

    # Test with larger n to get r5=0.0
    n_large = 10
    dot_large = torch.ones(n_large, n_large, dtype=torch.float32)
    dot_large.fill_diagonal_(0.0)

    metrics_large, other_large = compute_retrieval_cosine(dot_large)

    assert metrics_large["r1"] == 0.0
    assert metrics_large["r5"] == 0.0  # Now r5 should be 0.0 since ranks are 9
    assert metrics_large["medr"] == n_large
    assert metrics_large["meanr"] == n_large


def test_compute_retrieval_cosine_random():
    """Test with random similarity matrix"""
    n = 10
    torch.manual_seed(42)  # For reproducible results
    dot = torch.randn(n, n, dtype=torch.float32)

    metrics, other = compute_retrieval_cosine(dot)

    # Basic sanity checks
    assert 0.0 <= metrics["r1"] <= 1.0
    assert 0.0 <= metrics["r5"] <= 1.0
    assert metrics["r1"] <= metrics["r5"]  # r1 should be <= r5
    assert 1.0 <= metrics["medr"] <= n
    assert 1.0 <= metrics["meanr"] <= n
    assert metrics["n"] == n

    # Check shapes
    assert other["ranks"].shape == (n,)
    assert other["top1"].shape == (n,)
    assert torch.all(other["ranks"] >= 0)
    assert torch.all(other["ranks"] < n)
    assert torch.all(other["top1"] >= 0)
    assert torch.all(other["top1"] < n)


def test_compute_retrieval_cosine_mixed():
    """Test with mixed performance - some correct, some wrong"""
    n = 6
    # Create matrix where first 3 are correct, last 3 are wrong
    dot = torch.zeros(n, n, dtype=torch.float32)

    # First 3 rows: diagonal is highest
    dot[:3, :3] = torch.eye(3)
    dot[:3, 3:] = 0.5  # Lower similarity for off-diagonal

    # Last 3 rows: off-diagonal is highest
    dot[3:, 3:] = torch.ones(3, 3) - torch.eye(3)  # Off-diagonal highest
    dot[3:, :3] = 0.5  # Lower similarity for diagonal

    metrics, other = compute_retrieval_cosine(dot)

    # Should have 50% correct retrievals
    assert metrics["r1"] == 0.5  # 3 out of 6 correct
    assert metrics["r5"] == 0.5  # 3 out of 6 correct
    assert metrics["n"] == n

    # Check ranks: first 3 should be 0, last 3 should be > 0
    assert torch.all(other["ranks"][:3] == 0)
    assert torch.all(other["ranks"][3:] > 0)


def test_compute_retrieval_cosine_edge_cases():
    """Test edge cases"""
    # Single element
    dot = torch.tensor([[1.0]], dtype=torch.float32)
    metrics, other = compute_retrieval_cosine(dot)

    assert metrics["r1"] == 1.0
    assert metrics["r5"] == 1.0
    assert metrics["medr"] == 1.0
    assert metrics["meanr"] == 1.0
    assert metrics["n"] == 1
    assert other["ranks"][0] == 0
    assert other["top1"][0] == 0

    # Two elements, perfect retrieval
    dot = torch.eye(2, dtype=torch.float32)
    metrics, other = compute_retrieval_cosine(dot)

    assert metrics["r1"] == 1.0
    assert metrics["r5"] == 1.0
    assert metrics["medr"] == 1.0
    assert metrics["meanr"] == 1.0
    assert metrics["n"] == 2
