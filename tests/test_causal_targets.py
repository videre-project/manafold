"""Unit tests for Candidate C causal soft target generator and loss module."""

import unittest
import torch
from manafold.models.classifiers.causal_soft_targets import CausalSoftTargetGenerator, SoftTargetCrossEntropyLoss


class TestCausalTargets(unittest.TestCase):

    def test_causal_soft_target_generator_properties(self):
        num_classes = 4
        generator = CausalSoftTargetGenerator(num_classes=num_classes, alpha=0.2)
        
        targets = [0, 0, 1, 2, 3]
        deck_cards = [
            [101, 102, 103], # class 0
            [101, 102, 104], # class 0
            [101, 102, 105], # class 1 (shares cards 101, 102 with class 0)
            [201, 202],      # class 2
            [301, 302],      # class 3
        ]
        
        soft_matrix = generator.fit_from_historical_data(
            target_indices=targets,
            card_features_by_deck=deck_cards,
        )
        
        self.assertEqual(soft_matrix.shape, (num_classes, num_classes))
        # Check valid probability row sums
        row_sums = torch.sum(soft_matrix, dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(num_classes), atol=1e-5))
        
        # Class 0 should have non-zero probability for Class 1 due to card sharing (101, 102)
        self.assertGreater(soft_matrix[0, 1].item(), 0.0)
        # Class 0 and Class 3 share no cards, so soft_matrix[0, 3] should be 0.0
        self.assertEqual(soft_matrix[0, 3].item(), 0.0)

    def test_soft_target_cross_entropy_loss(self):
        loss_fn = SoftTargetCrossEntropyLoss(reduction="mean")
        logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
        soft_targets = torch.tensor([[0.8, 0.2, 0.0], [0.1, 0.8, 0.1]])
        
        loss = loss_fn(logits, soft_targets)
        expected = -(
            soft_targets * torch.log_softmax(logits, dim=-1)
        ).sum(dim=-1).mean()
        self.assertTrue(torch.allclose(loss, expected, atol=1e-7))

    def test_get_soft_targets_selects_rows_from_the_fitted_matrix(self):
        num_classes = 3
        generator = CausalSoftTargetGenerator(num_classes=num_classes, alpha=0.1)
        
        past_targets = [0, 1]
        past_decks = [[1, 2], [2, 3]]
        
        soft_matrix = generator.fit_from_historical_data(
            target_indices=past_targets,
            card_features_by_deck=past_decks,
        )
        
        batch = torch.tensor([0, 1])
        selected_targets = generator.get_soft_targets(batch)
        
        self.assertTrue(torch.equal(selected_targets, soft_matrix[batch]))


if __name__ == "__main__":
    unittest.main()
