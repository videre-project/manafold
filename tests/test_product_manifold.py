import unittest
import torch

from manafold.models.classifiers.product_manifold import ProductManifoldEmbedding, PoincaréHyperbolicSpace


class TestProductManifold(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.manifold = ProductManifoldEmbedding(in_dim=64, euc_dim=32, hyp_dim=32)

    def test_forward_output_shapes_and_bounds(self):
        x = torch.randn(8, 64)
        z_euc, z_hyp = self.manifold(x)
        self.assertEqual(z_euc.shape, (8, 32))
        self.assertEqual(z_hyp.shape, (8, 32))
        # Norm of z_hyp in Poincaré ball must be strictly < 1.0
        norms = torch.linalg.vector_norm(z_hyp, dim=-1)
        self.assertTrue((norms < 1.0).all())

    def test_poincare_distance_nonnegative(self):
        u = torch.tensor([[0.1, 0.2], [0.3, -0.4]])
        v = torch.tensor([[0.1, 0.2], [0.5, 0.1]])
        dist = PoincaréHyperbolicSpace.distance(u, v)
        self.assertAlmostEqual(dist[0].item(), 0.0, places=4)
        self.assertGreater(dist[1].item(), 0.0)

    def test_tree_loss_computation(self):
        x = torch.randn(4, 64)
        _, z_hyp = self.manifold(x)
        targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        families = torch.tensor([0, 0, 0, 0], dtype=torch.long)
        loss = self.manifold.compute_tree_loss(z_hyp, targets, families)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_tree_loss_has_finite_gradients_at_coincident_points(self):
        x = torch.randn(4, 64, requires_grad=True)
        _, z_hyp = self.manifold(x)
        targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        families = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        loss = self.manifold.compute_tree_loss(z_hyp, targets, families)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
