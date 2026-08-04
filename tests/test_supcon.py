"""Unit tests for Supervised Set-Level Contrastive Loss (SupConLoss)."""

import unittest
import torch

from manafold.models.classifiers.supervised_contrastive_loss import SupConLoss


class TestSupConLoss(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(13)

    def test_supcon_loss_output(self):
        loss_fn = SupConLoss(temperature=0.07)
        batch_size = 4
        dim = 16
        
        # 2 classes, 2 samples per class
        features = torch.randn(batch_size, dim)
        labels = torch.tensor([0, 0, 1, 1])
        
        loss = loss_fn(features, labels)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_supcon_loss_with_partial_views(self):
        loss_fn = SupConLoss(temperature=0.07)
        batch_size = 4
        dim = 16
        
        features_full = torch.randn(batch_size, dim)
        features_partial = torch.randn(batch_size, dim)
        labels = torch.tensor([0, 1, 0, 1])
        
        loss = loss_fn(features_full, labels, partial_features=features_partial)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_supcon_perfect_clustering(self):
        """Identical features in same class should yield lower loss than orthogonal ones."""
        loss_fn = SupConLoss(temperature=0.07)
        
        # Perfect alignment
        v1 = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        labels = torch.tensor([0, 0, 1, 1])
        
        loss_perfect = loss_fn(v1, labels)
        
        # Random noise
        v2 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        loss_mixed = loss_fn(v2, labels)
        
        self.assertLess(loss_perfect.item(), loss_mixed.item())


if __name__ == "__main__":
    unittest.main()
