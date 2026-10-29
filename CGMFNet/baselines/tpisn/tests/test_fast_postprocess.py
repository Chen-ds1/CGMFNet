import unittest
import numpy as np
from sklearn.metrics import adjusted_rand_score
from tpisn.postprocess import cluster_embeddings, fit_line
from tpisn.fast_postprocess import cluster_embeddings_fast, fit_line_fast


class FastTests(unittest.TestCase):
    def test_separated_rows_and_noise(self):
        rng = np.random.default_rng(123)
        points = np.concatenate([rng.normal(loc=c, scale=.08, size=(350,4)) for c in (0,4,8)] +
                                [rng.uniform(20,30,(10,4))])
        emb = points.T[:, None, :]
        mask = np.ones((1,len(points)),bool)
        exact = cluster_embeddings(emb,mask)
        fast = cluster_embeddings_fast(emb,mask,max_anchors=128)
        self.assertEqual(adjusted_rand_score(exact.ravel(),fast.ravel()),1.)
        self.assertTrue((fast[0,-10:]==0).all())

    def test_wls_matches_original_objective(self):
        rng = np.random.default_rng(17)
        for k in [-2.,0.,.3,2.]:
            y = rng.uniform(0,255,2000)
            x = k*y + 80 + rng.normal(0,3,len(y))
            x[::25] += 100
            points=np.column_stack([x,y])
            a,b=fit_line(points),fit_line_fast(points)
            self.assertAlmostEqual(a['k'],b['k'],places=8)
            self.assertAlmostEqual(a['b'],b['b'],places=6)
            self.assertEqual(a['iterations'],b['iterations'])
            self.assertEqual(a['converged'],b['converged'])

    def test_empty_small_dense_and_degenerate(self):
        emb=np.zeros((4,16,32));mask=np.ones((16,32),bool)
        self.assertFalse(cluster_embeddings_fast(emb,~mask).any())
        self.assertFalse(cluster_embeddings_fast(emb,mask,min_samples=1000).any())
        self.assertTrue((cluster_embeddings_fast(emb,mask)==1).all())
        self.assertIsNone(fit_line_fast([[1,1],[3,1]]))

    def test_repeatability_and_input_validation(self):
        rng=np.random.default_rng(18);emb=rng.normal(size=(4,32,32));mask=np.ones((32,32),bool)
        a=cluster_embeddings_fast(emb,mask,min_samples=10,max_anchors=128)
        b=cluster_embeddings_fast(emb,mask,min_samples=10,max_anchors=128)
        np.testing.assert_array_equal(a,b)
        with self.assertRaises(ValueError):cluster_embeddings_fast(emb,mask,max_anchors=0)


if __name__=='__main__':
    unittest.main()
