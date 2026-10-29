import unittest
import numpy as np
import torch
from tpisn.model import TPISN
from tpisn.losses import discriminative_loss, focal_loss, total_loss
from tpisn.data import labelme_mask
from tpisn.postprocess import cluster_embeddings, fit_line, angle_error
from tpisn.inference import original_lines
from evaluate import match_lines


class CoreTests(unittest.TestCase):
    def test_analytical_losses(self):

        emb = torch.tensor([[[[0., 2.]]]], requires_grad=True)
        ids = torch.tensor([[[1, 2]]])
        var, dist = discriminative_loss(emb, ids)
        self.assertAlmostEqual(var.item(), 0)
        self.assertAlmostEqual(dist.item(), 1)
        (var + dist).backward()
        self.assertTrue(torch.isfinite(emb.grad).all())
        focal = focal_loss(torch.zeros(1, 1, 1, 2), torch.tensor([[[[0, 1]]]]))
        self.assertAlmostEqual(focal.item(), np.log(2) * .25 * .5, places=6)

    def test_empty_and_single_instance(self):
        for label in (0, 1):
            emb = torch.ones(1, 4, 5, 7, requires_grad=True)
            var, dist = discriminative_loss(emb, torch.full((1, 5, 7), label))
            self.assertEqual((var + dist).item(), 0)
            (var + dist).backward()
            self.assertTrue(torch.isfinite(emb.grad).all())

    def test_annotation_groups_and_resize(self):
        ann = dict(imageWidth=1024, imageHeight=512, shapes=[
            dict(shape_type='linestrip', points=[[200, 0], [200, 250]], group_id=8),
            dict(shape_type='linestrip', points=[[200, 252], [200, 510]], group_id=8),
            dict(shape_type='line', points=[[800, 0], [800, 510]], group_id=None)])
        mask, lines = labelme_mask(ann)
        self.assertEqual(mask[10, 100], 1)
        self.assertEqual(mask[200, 400], 2)
        self.assertEqual(lines[0]['instance_id'], lines[1]['instance_id'])

    def test_dbscan_density_and_noise(self):
        emb = np.zeros((2, 1, 405), np.float32)
        emb[:, :, 200:400] = 4
        emb[:, :, 400:] = 20
        foreground = np.ones((1, 405), bool)
        result = cluster_embeddings(emb, foreground)
        self.assertEqual(len(np.unique(result)), 3)
        self.assertTrue((result[:, 400:] == 0).all())
        self.assertTrue((result[:, :200] > 0).all())
        self.assertFalse(cluster_embeddings(emb, ~foreground).any())

    def test_wls_and_scaling(self):
        y = np.arange(256)
        points = np.column_stack([0.25*y + 40, y])
        result = fit_line(points)
        self.assertAlmostEqual(result['k'], .25, places=6)
        noisy = np.vstack([points, [[400, 40], [450, 80], [470, 220]]])
        self.assertLess(angle_error(fit_line(noisy)['k'], .25), .2)
        scaled = original_lines([result], 1024, 256)[0]
        self.assertAlmostEqual(scaled['k'], .5)
        self.assertAlmostEqual(scaled['b'], 80)
        self.assertIsNone(fit_line([[1, 1], [2, 1]]))

    def test_dbscan_matches_reference(self):
        from sklearn.cluster import DBSCAN
        rng = np.random.default_rng(7)
        points = np.concatenate([rng.normal(loc=c, scale=.15, size=(80, 4)) for c in (0, 3, 6)] +
                                [rng.uniform(-10, 10, size=(10, 4))])
        expected = DBSCAN(eps=.5, min_samples=10).fit_predict(points)
        actual = cluster_embeddings(points.T[:, None, :], np.ones((1, len(points)), bool), .5, 10)[0]
        np.testing.assert_array_equal(actual == 0, expected == -1)
        np.testing.assert_array_equal(actual[:, None] == actual[None, :], expected[:, None] == expected[None, :])

    def test_matching_rejects_missed_rows(self):
        self.assertEqual(match_lines([dict(k=0, b=200)], [dict(k=0, b=10)]), [])
        self.assertEqual(match_lines([dict(k=0, b=10)], [dict(k=0, b=10)]), [0.0])

    def test_forward_backward_both_heads(self):
        torch.set_num_threads(2)
        model = TPISN(channels=32).train()
        x = torch.rand(2, 3, 64, 128)
        ids = torch.zeros(2, 64, 128, dtype=torch.long)
        ids[:, :, 20:30] = 1
        ids[:, :, 90:100] = 2
        output = model(x)
        self.assertEqual(tuple(output['embeddings'].shape), (2, 4, 64, 128))
        loss, _ = total_loss(output, ids)
        loss.backward()
        for branch in (model.backbone, model.context, model.segmentation, model.embedding):
            gradients = [p.grad for p in branch.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        model.eval()
        with torch.no_grad():
            output = model(torch.rand(1, 3, 256, 512))
        self.assertEqual(tuple(output['logits'].shape), (1, 1, 256, 512))


if __name__ == '__main__':
    unittest.main()
