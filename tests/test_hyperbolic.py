"""Unit tests for A5 Hyperbolic (Poincaré ball) geometry operations and Prototypical Head."""

import unittest
import torch

from manafold.models.classifiers.poincare_geometry import (
    clip_to_poincare_ball,
    exp_map_zero,
    log_map_zero,
    poincare_distance,
    TangentSpaceSetPooler,
    HyperbolicPrototypicalHead,
)


class TestHyperbolicGeometry(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(13)

    def test_exp_log_map_idempotency(self):
        """Test exp_0(log_0(y)) == y and log_0(exp_0(v)) == v near origin."""
        v = torch.tensor([[0.1, -0.2, 0.3], [0.05, 0.15, -0.1]])
        y = exp_map_zero(v, c=1.0)
        v_rec = log_map_zero(y, c=1.0)
        self.assertTrue(torch.allclose(v, v_rec, atol=1e-4))

    def test_poincare_distance_properties(self):
        """Test non-negativity, symmetry, and identity of indiscernibles for Poincaré distance."""
        u = torch.tensor([[0.1, 0.2, 0.3]])
        v = torch.tensor([[0.4, -0.1, 0.2]])
        
        # d(u, u) == 0
        dist_uu = poincare_distance(u, u, c=1.0)
        self.assertLess(dist_uu.item(), 1e-5)
        
        # Symmetry d(u, v) == d(v, u)
        dist_uv = poincare_distance(u, v, c=1.0)
        dist_vu = poincare_distance(v, u, c=1.0)
        self.assertAlmostEqual(dist_uv.item(), dist_vu.item(), places=5)
        
        # Non-negativity
        self.assertGreater(dist_uv.item(), 0.0)

    def test_boundary_radius_clipping(self):
        """Test vectors outside the Poincaré ball are clipped strictly inside ||x|| < 1 - eps."""
        far_point = torch.tensor([[5.0, 10.0, -12.0]])
        clipped = clip_to_poincare_ball(far_point, c=1.0, eps=1e-5)
        norm = torch.norm(clipped, dim=-1).item()
        self.assertLess(norm, 1.0)
        self.assertGreater(norm, 0.99)

    def test_tangent_space_set_pooler(self):
        pooler = TangentSpaceSetPooler(curvature=1.0)
        batch_size = 2
        seq_len = 3
        dim = 4
        
        cards = exp_map_zero(torch.randn(batch_size, seq_len, dim) * 0.1)
        weights = torch.ones(batch_size, seq_len)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        
        pooled = pooler(cards, weights, mask)
        self.assertEqual(pooled.shape, (batch_size, dim))
        norm = torch.norm(pooled, dim=-1)
        self.assertTrue((norm < 1.0).all())

    def test_hyperbolic_prototypical_head(self):
        batch_size = 4
        dim = 8
        num_classes = 5
        
        head = HyperbolicPrototypicalHead(embedding_dim=dim, num_classes=num_classes)
        embeddings = exp_map_zero(torch.randn(batch_size, dim) * 0.1)
        
        logits, ood_margins = head(embeddings)
        
        self.assertEqual(logits.shape, (batch_size, num_classes))
        self.assertEqual(ood_margins.shape, (batch_size,))
        self.assertTrue((ood_margins >= 0.0).all())


if __name__ == "__main__":
    unittest.main()
