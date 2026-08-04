import unittest
import torch
from dataclasses import dataclass

from manafold.models.classifiers.pmi_anchor_gate import PMIAnchorGate


@dataclass(frozen=True)
class DummyToken:
    card_idx: int


@dataclass(frozen=True)
class DummyExample:
    target_label_id: str
    tokens: tuple[DummyToken, ...]


class TestPMIAnchorGate(unittest.TestCase):
    def setUp(self):
        self.labels = ["archetype_a", "archetype_b"]
        # Create dummy examples where card 0 is unique to archetype_a, card 1 is unique to archetype_b, card 2 is common
        exs = []
        for _ in range(10):
            exs.append(DummyExample("archetype_a", (DummyToken(0), DummyToken(2))))
            exs.append(DummyExample("archetype_b", (DummyToken(1), DummyToken(2))))
        self.examples = exs
        self.gate = PMIAnchorGate(
            card_count=5,
            labels=self.labels,
            examples=self.examples,
            device="cpu",
        )

    def test_pmi_scores_computed(self):
        # Anchor scores for unique cards (0, 1) should be higher than common card (2)
        scores = self.gate.anchor_scores
        self.assertGreater(scores[0].item(), scores[2].item())
        self.assertGreater(scores[1].item(), scores[2].item())

    def test_forward_shape(self):
        card_indices = torch.tensor([[0, 2], [1, 2]], dtype=torch.long)
        bias = self.gate(card_indices)
        self.assertEqual(bias.shape, card_indices.shape)

    def test_refit_replaces_scores_from_the_supplied_split(self):
        initial_scores = self.gate.anchor_scores.clone()
        self.gate.fit([
            DummyExample("archetype_a", (DummyToken(3),)),
            DummyExample("archetype_b", (DummyToken(3),)),
        ])

        self.assertFalse(torch.equal(initial_scores, self.gate.anchor_scores))
        self.assertEqual(0.0, self.gate.anchor_scores[0].item())
        self.assertEqual(0.0, self.gate.anchor_scores[1].item())


if __name__ == "__main__":
    unittest.main()
