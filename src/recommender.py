"""
Content-based movie recommendation engine.

Every movie is turned into a weighted TF-IDF vector built from its genres,
director, cast and plot overview. Similarity between two movies is the cosine
of the angle between their vectors. Only the Python standard library is used.
"""

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "by", "for",
    "with", "from", "as", "is", "are", "was", "his", "her", "their", "its",
    "he", "she", "they", "him", "them", "who", "that", "this", "must", "into",
    "after", "over", "than", "then", "while", "one", "two", "three", "not",
}

# Feature weights: higher means the field influences similarity more.
WEIGHT_GENRE = 3
WEIGHT_DIRECTOR = 2
WEIGHT_CAST = 2
WEIGHT_OVERVIEW = 1


@dataclass
class Movie:
    title: str
    year: int
    genres: List[str]
    directors: List[str]
    cast: List[str]
    rating: float
    overview: str
    vector: Dict[str, float] = field(default_factory=dict, repr=False)
    norm: float = field(default=0.0, repr=False)

    def summary(self) -> str:
        return (f"{self.title} ({self.year}) | {', '.join(self.genres)} | "
                f"Rating: {self.rating}")


def tokenize(text: str) -> List[str]:
    """Lowercase, split into alphanumeric words and drop stop words."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in STOPWORDS and len(t) > 1]


def _name_token(prefix: str, name: str) -> str:
    """Turn 'Christopher Nolan' into 'dir_christopher_nolan' (one token)."""
    return prefix + "_" + "_".join(re.findall(r"[a-z0-9]+", name.lower()))


class MovieRecommender:
    """Loads a movie CSV and provides recommendation / search methods."""

    def __init__(self, csv_path: Optional[str] = None):
        if csv_path is None:
            csv_path = Path(__file__).resolve().parent.parent / "data" / "movies.csv"
        self.csv_path = Path(csv_path)
        self.movies: List[Movie] = []
        self.idf: Dict[str, float] = {}
        self._load()
        self._build_vectors()

    # ------------------------------------------------------------------ data
    def _load(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")
        with open(self.csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.movies.append(Movie(
                    title=row["title"].strip(),
                    year=int(row["year"]),
                    genres=[g.strip() for g in row["genres"].split("|") if g.strip()],
                    directors=[d.strip() for d in row["director"].split("|") if d.strip()],
                    cast=[c.strip() for c in row["cast"].split("|") if c.strip()],
                    rating=float(row["rating"]),
                    overview=row["overview"].strip(),
                ))
        if not self.movies:
            raise ValueError("The dataset is empty.")

    def _raw_counts(self, m: Movie) -> Counter:
        counts: Counter = Counter()
        for g in m.genres:
            counts["genre_" + g.lower()] += WEIGHT_GENRE
        for d in m.directors:
            counts[_name_token("dir", d)] += WEIGHT_DIRECTOR
        for c in m.cast:
            counts[_name_token("cast", c)] += WEIGHT_CAST
        for t in tokenize(m.overview):
            counts[t] += WEIGHT_OVERVIEW
        return counts

    def _build_vectors(self) -> None:
        raw = [self._raw_counts(m) for m in self.movies]
        n_docs = len(self.movies)
        doc_freq: Counter = Counter()
        for counts in raw:
            doc_freq.update(counts.keys())
        self.idf = {t: math.log((1 + n_docs) / (1 + df)) + 1 for t, df in doc_freq.items()}
        for movie, counts in zip(self.movies, raw):
            vec = {t: (1 + math.log(c)) * self.idf[t] if c else 0.0
                   for t, c in counts.items()}
            movie.vector = vec
            movie.norm = math.sqrt(sum(v * v for v in vec.values()))

    # ------------------------------------------------------------ similarity
    @staticmethod
    def _cosine(a: Dict[str, float], na: float, b: Dict[str, float], nb: float) -> float:
        if na == 0 or nb == 0:
            return 0.0
        if len(a) > len(b):
            a, b = b, a
        dot = sum(v * b.get(t, 0.0) for t, v in a.items())
        return dot / (na * nb)

    def similarity(self, m1: Movie, m2: Movie) -> float:
        return self._cosine(m1.vector, m1.norm, m2.vector, m2.norm)

    # -------------------------------------------------------------- lookups
    def find_movie(self, title: str) -> Optional[Movie]:
        """Exact match -> substring match -> fuzzy match (typo tolerant)."""
        q = title.strip().lower()
        if not q:
            return None
        for m in self.movies:
            if m.title.lower() == q:
                return m
        subs = [m for m in self.movies if q in m.title.lower()]
        if subs:
            return sorted(subs, key=lambda m: len(m.title))[0]
        close = get_close_matches(q, [m.title.lower() for m in self.movies], n=1, cutoff=0.6)
        if close:
            return next(m for m in self.movies if m.title.lower() == close[0])
        return None

    def all_genres(self) -> List[str]:
        return sorted({g for m in self.movies for g in m.genres})

    # ------------------------------------------------------ recommendations
    def similar_to(self, title: str, n: int = 5) -> List[Tuple[Movie, float]]:
        """Top-n movies most similar to the given title."""
        base = self.find_movie(title)
        if base is None:
            raise LookupError(f"Movie '{title}' not found in the dataset.")
        scored = [(m, self.similarity(base, m)) for m in self.movies if m is not base]
        scored.sort(key=lambda x: (-x[1], -x[0].rating))
        return scored[:n]

    def recommend_for_liked(self, titles: Sequence[str], n: int = 5) -> List[Tuple[Movie, float]]:
        """Build a user profile from several liked movies and recommend new ones."""
        liked: List[Movie] = []
        for t in titles:
            m = self.find_movie(t)
            if m is None:
                raise LookupError(f"Movie '{t}' not found in the dataset.")
            if m not in liked:
                liked.append(m)
        if not liked:
            raise ValueError("Please provide at least one liked movie.")
        profile: Dict[str, float] = {}
        for m in liked:
            for t, v in m.vector.items():
                profile[t] = profile.get(t, 0.0) + v / m.norm
        pnorm = math.sqrt(sum(v * v for v in profile.values()))
        scored = [(m, self._cosine(profile, pnorm, m.vector, m.norm))
                  for m in self.movies if m not in liked]
        scored.sort(key=lambda x: (-x[1], -x[0].rating))
        return scored[:n]

    # --------------------------------------------------------------- filters
    def search(self, genre: Optional[str] = None, year_from: Optional[int] = None,
               year_to: Optional[int] = None, min_rating: Optional[float] = None,
               director: Optional[str] = None, actor: Optional[str] = None,
               limit: int = 10) -> List[Movie]:
        results = []
        for m in self.movies:
            if genre and genre.lower() not in [g.lower() for g in m.genres]:
                continue
            if year_from is not None and m.year < year_from:
                continue
            if year_to is not None and m.year > year_to:
                continue
            if min_rating is not None and m.rating < min_rating:
                continue
            if director and not any(director.lower() in d.lower() for d in m.directors):
                continue
            if actor and not any(actor.lower() in c.lower() for c in m.cast):
                continue
            results.append(m)
        results.sort(key=lambda m: (-m.rating, m.title))
        return results[:limit]

    def top_rated(self, n: int = 10, genre: Optional[str] = None) -> List[Movie]:
        return self.search(genre=genre, limit=n)
