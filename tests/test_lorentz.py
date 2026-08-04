"""Unit tests for Lorentz (Hyperboloid) manifold operations, Transformer blocks, and Prototypical Head."""

import unittest
import torch

from manafold.models.classifiers.lorentz import (
    minkowski_inner_product,
    project_to_lorentz,
    exp_map_lorentz_zero,
    log_map_lorentz_zero,
    lorentz_distance,
    HybridLorentzAttention,
    LorentzTransformerBlock,
    LorentzPrototypicalHead,
)


class TestLorentzGeometry(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(13)

    def test_hyperboloid_constraint(self):
        """Test <x, x>_L == -1/c for projected Lorentz points."""
        c = 1.0
        v_spatial = torch.randn(5, 4) * 0.5
        x_lorentz = exp_map_lorentz_zero(v_spatial, c=c)

        inner = minkowski_inner_product(x_lorentz, x_lorentz)
        expected = torch.full_like(inner, -1.0 / c)
        self.assertTrue(torch.allclose(inner, expected, atol=1e-4))

    def test_exp_log_map_lorentz_idempotency(self):
        """Test exp_0(log_0(x)) == x and log_0(exp_0(v)) == v in Lorentz space."""
        v = torch.tensor([[0.1, -0.2, 0.3], [0.05, 0.15, -0.1]])
        x_lorentz = exp_map_lorentz_zero(v, c=1.0)
        v_rec = log_map_lorentz_zero(x_lorentz, c=1.0)
        self.assertTrue(torch.allclose(v, v_rec, atol=1e-4))

    def test_lorentz_distance_properties(self):
        """Test non-negativity, symmetry, and zero distance for Lorentz points."""
        v1 = torch.tensor([[0.1, 0.2, 0.3]])
        v2 = torch.tensor([[0.4, -0.1, 0.2]])

        u = exp_map_lorentz_zero(v1, c=1.0)
        v = exp_map_lorentz_zero(v2, c=1.0)

        # d(u, u) == 0
        dist_uu = lorentz_distance(u, u, c=1.0)
        self.assertLess(dist_uu.item(), 1e-4)

        # Symmetry d(u, v) == d(v, u)
        dist_uv = lorentz_distance(u, v, c=1.0)
        dist_vu = lorentz_distance(v, u, c=1.0)
        self.assertAlmostEqual(dist_uv.item(), dist_vu.item(), places=4)

        # Non-negativity
        self.assertGreater(dist_uv.item(), 0.0)

    def test_lorentz_transformer_block(self):
        """Test full Lorentz Transformer block with residual + LN + FFN."""
        batch_size = 2
        seq_len = 3
        spatial_dim = 8

        block = LorentzTransformerBlock(spatial_dim=spatial_dim, ffn_dim=32)
        v_spatial = torch.randn(batch_size, seq_len, spatial_dim) * 0.1
        x_lorentz = exp_map_lorentz_zero(v_spatial)
        mask = torch.tensor([[True, True, False], [True, True, True]])

        out = block(x_lorentz, mask)
        self.assertEqual(out.shape, (batch_size, seq_len, spatial_dim + 1))

        # Check hyperboloid constraint on output
        inner = minkowski_inner_product(out, out)
        self.assertTrue(torch.allclose(inner, torch.full_like(inner, -1.0), atol=1e-3))

    def test_lorentz_prototypical_head(self):
        batch_size = 4
        spatial_dim = 8
        num_classes = 5

        head = LorentzPrototypicalHead(spatial_dim=spatial_dim, num_classes=num_classes)
        v_spatial = torch.randn(batch_size, spatial_dim) * 0.1
        x_lorentz = exp_map_lorentz_zero(v_spatial)

        logits, ood_margins = head(x_lorentz)

        self.assertEqual(logits.shape, (batch_size, num_classes))
        self.assertEqual(ood_margins.shape, (batch_size,))
        self.assertTrue((ood_margins >= 0.0).all())

    def test_stacked_blocks_preserve_hyperboloid(self):
        """Two stacked blocks should still produce valid hyperboloid points."""
        batch_size = 2
        seq_len = 4
        spatial_dim = 8

        block1 = LorentzTransformerBlock(spatial_dim=spatial_dim)
        block2 = LorentzTransformerBlock(spatial_dim=spatial_dim)
        v = torch.randn(batch_size, seq_len, spatial_dim) * 0.1
        x = exp_map_lorentz_zero(v)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        out = block2(block1(x, mask), mask)
        inner = minkowski_inner_product(out, out)
        self.assertTrue(torch.allclose(inner, torch.full_like(inner, -1.0), atol=1e-3))


if __name__ == "__main__":
    unittest.main()
