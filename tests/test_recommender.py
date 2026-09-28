import unittest
from pathlib import Path

from src.recommender import MovieRecommender, tokenize

DATA = Path(__file__).resolve().parent.parent / "data" / "movies.csv"


class TestRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = MovieRecommender(DATA)

    def test_dataset_loaded(self):
        self.assertGreaterEqual(len(self.rec.movies), 50)

    def test_tokenize_removes_stopwords(self):
        self.assertEqual(tokenize("The Matrix and a hacker"), ["matrix", "hacker"])

    def test_find_movie_exact_and_fuzzy(self):
        self.assertEqual(self.rec.find_movie("inception").title, "Inception")
        self.assertEqual(self.rec.find_movie("Intersteller").title, "Interstellar")
        self.assertIsNone(self.rec.find_movie("zzzzzzzz qqqq"))

    def test_similar_excludes_self_and_sorted(self):
        res = self.rec.similar_to("Inception", 5)
        self.assertEqual(len(res), 5)
        self.assertNotIn("Inception", [m.title for m, _ in res])
        scores = [s for _, s in res]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_similar_is_sensible(self):
        titles = [m.title for m, _ in self.rec.similar_to("Inception", 5)]
        self.assertTrue({"Interstellar", "The Prestige", "The Dark Knight"} & set(titles))

    def test_self_similarity_is_one(self):
        m = self.rec.find_movie("Inception")
        self.assertAlmostEqual(self.rec.similarity(m, m), 1.0, places=6)

    def test_liked_profile_excludes_liked(self):
        res = self.rec.recommend_for_liked(["3 Idiots", "PK"], 5)
        titles = [m.title for m, _ in res]
        self.assertNotIn("3 Idiots", titles)
        self.assertNotIn("PK", titles)

    def test_unknown_movie_raises(self):
        with self.assertRaises(LookupError):
            self.rec.similar_to("qwertyuiop asdfgh")

    def test_search_filters(self):
        res = self.rec.search(genre="Horror", min_rating=7.6)
        self.assertTrue(res)
        self.assertTrue(all("Horror" in m.genres and m.rating >= 7.6 for m in res))
        res = self.rec.search(director="Nolan", year_from=2010)
        self.assertTrue(all(m.year >= 2010 for m in res))

    def test_top_rated_order(self):
        res = self.rec.top_rated(5)
        self.assertEqual([m.rating for m in res], sorted([m.rating for m in res], reverse=True))


if __name__ == "__main__":
    unittest.main()
