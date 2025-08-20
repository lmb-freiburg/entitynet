import pytest
import torch

from entitynet.tasks.contrastive_task_multitext import (
    compute_retrieval_cosine_multitext_i2t,
    compute_retrieval_cosine_multitext_t2i,
)


def test_compute_retrieval_cosine_multitext_i2t_all_correct():
    """Test image-to-text with perfect retrieval - all images find their correct texts first"""
    n_images = 3
    n_texts_per_image = 2

    # Create similarity matrix where each image has highest similarity with its own texts
    # dot shape: (n_images, n_images * n_texts_per_image) = (3, 6)
    dot = torch.zeros(n_images, n_images * n_texts_per_image, dtype=torch.float32)

    # Image 0: highest similarity with texts 0,1 (its own texts)
    dot[0, 0] = 1.0  # image 0 with text 0
    dot[0, 1] = 1.0  # image 0 with text 1
    dot[0, 2:] = 0.5  # lower similarity with other texts

    # Image 1: highest similarity with texts 2,3 (its own texts)
    dot[1, 2] = 1.0  # image 1 with text 2
    dot[1, 3] = 1.0  # image 1 with text 3
    dot[1, [0, 1, 4, 5]] = 0.5  # lower similarity with other texts

    # Image 2: highest similarity with texts 4,5 (its own texts)
    dot[2, 4] = 1.0  # image 2 with text 4
    dot[2, 5] = 1.0  # image 2 with text 5
    dot[2, :4] = 0.5  # lower similarity with other texts

    # Indices mapping images to their text indices
    img2txt_indices = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)

    metrics, other = compute_retrieval_cosine_multitext_i2t(dot, img2txt_indices)

    # All images should find their correct texts at rank 0
    assert metrics["r1"] == 1.0  # 100% recall@1
    assert metrics["r5"] == 1.0  # 100% recall@5
    assert metrics["r10"] == 1.0  # 100% recall@10
    assert metrics["r20"] == 1.0  # 100% recall@20
    assert metrics["r50"] == 1.0  # 100% recall@50
    assert metrics["medr"] == 1.0  # median rank should be 1
    assert metrics["meanr"] == 1.0  # mean rank should be 1
    assert metrics["n"] == n_images

    # All ranks should be 0 (perfect retrieval)
    assert torch.all(other["ranks"] == 0)


def test_compute_retrieval_cosine_multitext_i2t_all_wrong():
    """Test image-to-text with worst retrieval - all images find correct texts last"""
    n_images = 3
    n_texts_per_image = 2

    # Create similarity matrix where each image has lowest similarity with its own texts
    dot = torch.ones(n_images, n_images * n_texts_per_image, dtype=torch.float32)

    # Set diagonal blocks to lowest similarity
    dot[0, [0, 1]] = 0.0  # image 0 with its texts
    dot[1, [2, 3]] = 0.0  # image 1 with its texts
    dot[2, [4, 5]] = 0.0  # image 2 with its texts

    img2txt_indices = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)

    metrics, other = compute_retrieval_cosine_multitext_i2t(dot, img2txt_indices)
    print("i2t all wrong ranks:", other["ranks"])
    # All images should find their correct texts at worst rank
    assert metrics["r1"] == 0.0  # 0% recall@1
    # For n=6, r5 should be 1.0 since all ranks are 5 and r5 counts ranks < 5
    assert metrics["r5"] == 1.0  # 100% recall@5 (all ranks are 5, which is < 5)
    assert metrics["r10"] == 1.0  # 100% recall@10
    assert metrics["r20"] == 1.0  # 100% recall@20
    assert metrics["r50"] == 1.0  # 100% recall@50
    assert metrics["medr"] == 5.0  # median rank should be 5.0
    assert metrics["meanr"] == 5.0  # mean rank should be 5.0
    assert metrics["n"] == n_images

    # All ranks should be worst possible (n_images * n_texts_per_image - 1)
    assert torch.all(
        other["ranks"] == 4
    )  # worst rank is 4 (0-indexed, so 5th position in 6-element array)

    # Test with larger n to get r5=0.0
    n_images_large = 5
    n_texts_per_image_large = 2
    dot_large = torch.ones(
        n_images_large, n_images_large * n_texts_per_image_large, dtype=torch.float32
    )

    # Set diagonal blocks to lowest similarity
    for i in range(n_images_large):
        start_idx = i * n_texts_per_image_large
        end_idx = (i + 1) * n_texts_per_image_large
        dot_large[i, start_idx:end_idx] = 0.0

    img2txt_indices_large = torch.arange(n_images_large * n_texts_per_image_large).reshape(
        n_images_large, n_texts_per_image_large
    )

    metrics_large, other_large = compute_retrieval_cosine_multitext_i2t(
        dot_large, img2txt_indices_large
    )

    assert metrics_large["r1"] == 0.0
    assert metrics_large["r5"] == 0.0  # Now r5 should be 0.0 since ranks are 9
    assert metrics_large["medr"] == 9.0
    assert metrics_large["meanr"] == 9.0


def test_compute_retrieval_cosine_multitext_i2t_mixed():
    """Test image-to-text with mixed performance - some correct, some wrong"""
    n_images = 4
    n_texts_per_image = 2

    # Create matrix where first 2 images are correct, last 2 are wrong
    dot = torch.zeros(n_images, n_images * n_texts_per_image, dtype=torch.float32)

    # First 2 images: correct texts have highest similarity
    dot[0, [0, 1]] = 1.0  # image 0 with texts 0,1
    dot[0, 2:] = 0.5
    dot[1, [2, 3]] = 1.0  # image 1 with texts 2,3
    dot[1, [0, 1, 4, 5, 6, 7]] = 0.5

    # Last 2 images: correct texts have lowest similarity
    dot[2, [4, 5]] = 0.0  # image 2 with texts 4,5 (lowest)
    dot[2, [0, 1, 2, 3, 6, 7]] = 1.0
    dot[3, [6, 7]] = 0.0  # image 3 with texts 6,7 (lowest)
    dot[3, :6] = 1.0

    img2txt_indices = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.long)

    metrics, other = compute_retrieval_cosine_multitext_i2t(dot, img2txt_indices)

    # Should have 50% correct retrievals
    assert metrics["r1"] == 0.5  # 2 out of 4 correct
    assert metrics["r5"] == 0.5  # 2 out of 4 correct
    assert metrics["n"] == n_images

    # Check ranks: first 2 should be 0, last 2 should be > 0
    assert torch.all(other["ranks"][:2] == 0)
    assert torch.all(other["ranks"][2:] > 0)


def test_compute_retrieval_cosine_multitext_t2i_all_correct():
    """Test text-to-image with perfect retrieval - all texts find their correct image first"""
    n_images = 3
    n_texts_per_image = 2

    # Create similarity matrix where each text has highest similarity with its correct image
    dot = torch.zeros(n_images, n_images * n_texts_per_image, dtype=torch.float32)

    # Text 0,1: highest similarity with image 0
    dot[0, 0] = 1.0  # image 0 with text 0
    dot[0, 1] = 1.0  # image 0 with text 1
    dot[1:, [0, 1]] = 0.5  # other images with texts 0,1

    # Text 2,3: highest similarity with image 1
    dot[1, 2] = 1.0  # image 1 with text 2
    dot[1, 3] = 1.0  # image 1 with text 3
    dot[[0, 2], [2, 3]] = 0.5  # other images with texts 2,3

    # Text 4,5: highest similarity with image 2
    dot[2, 4] = 1.0  # image 2 with text 4
    dot[2, 5] = 1.0  # image 2 with text 5
    dot[:2, [4, 5]] = 0.5  # other images with texts 4,5

    img2txt_indices = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)

    metrics, other = compute_retrieval_cosine_multitext_t2i(dot, img2txt_indices)

    # All texts should find their correct image at rank 0
    assert metrics["r1"] == 1.0  # 100% recall@1
    assert metrics["r5"] == 1.0  # 100% recall@5
    assert metrics["r10"] == 1.0  # 100% recall@10
    assert metrics["r20"] == 1.0  # 100% recall@20
    assert metrics["r50"] == 1.0  # 100% recall@50
    assert metrics["medr"] == 1.0  # median rank should be 1
    assert metrics["meanr"] == 1.0  # mean rank should be 1
    assert metrics["n"] == n_images

    # All ranks should be 0 (perfect retrieval)
    assert torch.all(other["ranks"] == 0)


def test_compute_retrieval_cosine_multitext_t2i_all_wrong():
    """Test text-to-image with worst retrieval - all texts find correct image last"""
    n_images = 3
    n_texts_per_image = 2

    # Create similarity matrix where each text has lowest similarity with its correct image
    dot = torch.ones(n_images, n_images * n_texts_per_image, dtype=torch.float32)

    # Set correct image-text pairs to lowest similarity
    dot[0, [0, 1]] = 0.0  # image 0 with its texts
    dot[1, [2, 3]] = 0.0  # image 1 with its texts
    dot[2, [4, 5]] = 0.0  # image 2 with its texts

    img2txt_indices = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)

    metrics, other = compute_retrieval_cosine_multitext_t2i(dot, img2txt_indices)

    # All texts should find their correct image at worst rank
    assert metrics["r1"] == 0.0  # 0% recall@1
    # For n=3, r5 should be 1.0 since all ranks are 2 and r5 counts ranks < 5
    assert metrics["r5"] == 1.0  # 100% recall@5 (all ranks are 2, which is < 5)
    assert metrics["r10"] == 1.0  # 100% recall@10
    assert metrics["r20"] == 1.0  # 100% recall@20
    assert metrics["r50"] == 1.0  # 100% recall@50
    assert metrics["medr"] == 3.0  # median rank should be 3.0
    assert metrics["meanr"] == 3.0  # mean rank should be 3.0
    assert metrics["n"] == n_images

    # All ranks should be worst possible (n_images - 1)
    assert torch.all(other["ranks"] == n_images - 1)

    # Test with larger n to get r5=0.0
    n_images_large = 6
    n_texts_per_image_large = 2
    dot_large = torch.ones(
        n_images_large, n_images_large * n_texts_per_image_large, dtype=torch.float32
    )

    # Set correct image-text pairs to lowest similarity
    for i in range(n_images_large):
        start_idx = i * n_texts_per_image_large
        end_idx = (i + 1) * n_texts_per_image_large
        dot_large[i, start_idx:end_idx] = 0.0

    img2txt_indices_large = torch.arange(n_images_large * n_texts_per_image_large).reshape(
        n_images_large, n_texts_per_image_large
    )

    metrics_large, other_large = compute_retrieval_cosine_multitext_t2i(
        dot_large, img2txt_indices_large
    )

    assert metrics_large["r1"] == 0.0
    assert metrics_large["r5"] == 0.0  # Now r5 should be 0.0 since ranks are 5
    assert metrics_large["medr"] == 6.0
    assert metrics_large["meanr"] == 6.0


def test_compute_retrieval_cosine_multitext_t2i_mixed():
    """Test text-to-image with mixed performance - some correct, some wrong"""
    n_images = 4
    n_texts_per_image = 2

    # Create matrix where first 4 texts are correct, last 4 are wrong
    dot = torch.zeros(n_images, n_images * n_texts_per_image, dtype=torch.float32)

    # First 4 texts: correct images have highest similarity
    dot[0, [0, 1]] = 1.0  # image 0 with texts 0,1
    dot[1:, [0, 1]] = 0.5
    dot[1, [2, 3]] = 1.0  # image 1 with texts 2,3
    dot[0, [2, 3]] = 0.5  # image 0 with texts 2,3
    dot[2:, [2, 3]] = 0.5  # images 2,3 with texts 2,3

    # Last 4 texts: correct images have lowest similarity
    dot[2, [4, 5]] = 0.0  # image 2 with texts 4,5 (lowest)
    dot[:2, [4, 5]] = 1.0  # images 0,1 with texts 4,5
    dot[3, [4, 5]] = 1.0  # image 3 with texts 4,5
    dot[3, [6, 7]] = 0.0  # image 3 with texts 6,7 (lowest)
    dot[:3, [6, 7]] = 1.0  # images 0,1,2 with texts 6,7

    img2txt_indices = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.long)

    metrics, other = compute_retrieval_cosine_multitext_t2i(dot, img2txt_indices)
    print("t2i mixed ranks:", other["ranks"])
    # Should have 50% correct retrievals
    assert metrics["r1"] == 0.5  # 4 out of 8 correct
    assert metrics["r5"] == 1.0  # all ranks < 5
    assert metrics["n"] == n_images

    # Check ranks: first 4 should be 0, last 4 should be > 0
    assert torch.all(other["ranks"][:4] == 0)
    assert torch.all(other["ranks"][4:] > 0)


def test_compute_retrieval_cosine_multitext_random():
    """Test with random similarity matrix"""
    n_images = 4
    n_texts_per_image = 3
    torch.manual_seed(42)  # For reproducible results

    # Random similarity matrix
    dot = torch.randn(n_images, n_images * n_texts_per_image, dtype=torch.float32)

    img2txt_indices = torch.arange(n_images * n_texts_per_image).reshape(
        n_images, n_texts_per_image
    )

    # Test i2t
    metrics_i2t, other_i2t = compute_retrieval_cosine_multitext_i2t(dot, img2txt_indices)

    # Basic sanity checks for i2t
    assert 0.0 <= metrics_i2t["r1"] <= 1.0
    assert 0.0 <= metrics_i2t["r5"] <= 1.0
    assert metrics_i2t["r1"] <= metrics_i2t["r5"]
    assert 1.0 <= metrics_i2t["medr"] <= n_images * n_texts_per_image
    assert 1.0 <= metrics_i2t["meanr"] <= n_images * n_texts_per_image
    assert metrics_i2t["n"] == n_images

    # Check shapes for i2t
    assert other_i2t["ranks"].shape == (n_images,)
    assert torch.all(other_i2t["ranks"] >= 0)
    assert torch.all(other_i2t["ranks"] < n_images * n_texts_per_image)

    # Test t2i
    metrics_t2i, other_t2i = compute_retrieval_cosine_multitext_t2i(dot, img2txt_indices)

    # Basic sanity checks for t2i
    assert 0.0 <= metrics_t2i["r1"] <= 1.0
    assert 0.0 <= metrics_t2i["r5"] <= 1.0
    assert metrics_t2i["r1"] <= metrics_t2i["r5"]
    assert 1.0 <= metrics_t2i["medr"] <= n_images
    assert 1.0 <= metrics_t2i["meanr"] <= n_images
    assert metrics_t2i["n"] == n_images

    # Check shapes for t2i
    assert other_t2i["ranks"].shape == (n_images * n_texts_per_image,)
    assert torch.all(other_t2i["ranks"] >= 0)
    assert torch.all(other_t2i["ranks"] < n_images)


def test_compute_retrieval_cosine_multitext_edge_cases():
    """Test edge cases"""
    # Single image, single text
    n_images = 1
    n_texts_per_image = 1

    dot = torch.tensor([[1.0]], dtype=torch.float32)
    img2txt_indices = torch.tensor([[0]], dtype=torch.long)

    # Test i2t
    metrics_i2t, other_i2t = compute_retrieval_cosine_multitext_i2t(dot, img2txt_indices)
    assert metrics_i2t["r1"] == 1.0
    assert metrics_i2t["r5"] == 1.0
    assert metrics_i2t["medr"] == 1.0
    assert metrics_i2t["meanr"] == 1.0
    assert metrics_i2t["n"] == n_images
    assert other_i2t["ranks"][0] == 0

    # Test t2i
    metrics_t2i, other_t2i = compute_retrieval_cosine_multitext_t2i(dot, img2txt_indices)
    assert metrics_t2i["r1"] == 1.0
    assert metrics_t2i["r5"] == 1.0
    assert metrics_t2i["medr"] == 1.0
    assert metrics_t2i["meanr"] == 1.0
    assert metrics_t2i["n"] == n_images
    assert other_t2i["ranks"][0] == 0

    # Two images, two texts each - need to create proper dot matrix
    n_images = 2
    n_texts_per_image = 2

    # Create dot matrix with correct shape (2, 4)
    dot = torch.zeros(n_images, n_images * n_texts_per_image, dtype=torch.float32)
    # Set diagonal elements to highest similarity
    dot[0, 0] = 1.0  # image 0 with text 0
    dot[0, 1] = 1.0  # image 0 with text 1
    dot[1, 2] = 1.0  # image 1 with text 2
    dot[1, 3] = 1.0  # image 1 with text 3
    # Set off-diagonal elements to lower similarity
    dot[0, [2, 3]] = 0.5
    dot[1, [0, 1]] = 0.5

    img2txt_indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

    # Test i2t
    metrics_i2t, other_i2t = compute_retrieval_cosine_multitext_i2t(dot, img2txt_indices)
    assert metrics_i2t["r1"] == 1.0
    assert metrics_i2t["r5"] == 1.0
    assert metrics_i2t["medr"] == 1.0
    assert metrics_i2t["meanr"] == 1.0
    assert metrics_i2t["n"] == n_images

    # Test t2i
    metrics_t2i, other_t2i = compute_retrieval_cosine_multitext_t2i(dot, img2txt_indices)
    assert metrics_t2i["r1"] == 1.0
    assert metrics_t2i["r5"] == 1.0
    assert metrics_t2i["medr"] == 1.0
    assert metrics_t2i["meanr"] == 1.0
    assert metrics_t2i["n"] == n_images
