# Project Report

## Movie Recommendation System using Content-Based Filtering

| | |
|---|---|
| **Student Name** | `<YOUR NAME>` |
| **Registration No.** | `<YOUR REG NO>` |
| **Course** | `<COURSE NAME & CODE>` |
| **Platform** | VITyarthi, VIT |
| **Faculty** | `<FACULTY NAME>` |
| **Date** | `<DATE>` |
| **GitHub Repository** | `https://github.com/<your-username>/movie-recommendation-system` |

---

## Table of Contents
1. Abstract
2. Introduction
3. Problem Statement
4. Objectives
5. Scope and Limitations
6. Literature Background
7. System Requirements
8. System Design
9. Methodology and Algorithm
10. Implementation
11. Testing
12. Results and Discussion
13. Challenges Faced
14. Future Enhancements
15. Conclusion
16. References

---

## 1. Abstract
This project presents a Movie Recommendation System that suggests films based on their content. Every movie is described by its genres, director, cast and plot summary, converted into a TF-IDF weighted vector, and compared with other movies using cosine similarity. The system supports single-movie recommendations, multi-movie taste profiles and rich search filters through a command-line interface. It is implemented in pure Python without external libraries, is covered by automated unit tests and is packaged for GitHub. Manual evaluation shows sensible outputs, for example *Inception* returns *The Prestige*, *The Dark Knight* and *Interstellar*, which share director, genre and themes.

## 2. Introduction
Streaming platforms host thousands of titles, and viewers often spend more time choosing than watching. Recommendation systems address this by filtering large catalogues into a short personalised list. The two dominant approaches are:

- **Collaborative filtering**: recommends items liked by similar users.
- **Content-based filtering**: recommends items whose attributes resemble those the user liked.

This project implements the content-based approach because it works without user history, is explainable ("recommended because it shares genre and director") and is simple enough to build from first principles.

## 3. Problem Statement
Given a catalogue of movies and one or more movies a user likes, produce a ranked list of other movies the user is most likely to enjoy, without relying on other users' rating data.

## 4. Objectives
1. Design a feature representation for movies.
2. Implement TF-IDF and cosine similarity from scratch.
3. Provide single-title and multi-title recommendations.
4. Provide search filters and top-rated lists.
5. Handle user errors such as typos and unknown titles gracefully.
6. Deliver a tested, documented, GitHub-ready project.

## 5. Scope and Limitations
**In scope:** offline dataset of 50 movies, CLI interface, content-based filtering.

**Limitations:**
- Small dataset, so variety of recommendations is limited.
- No personalisation across sessions (no user accounts).
- Plot text is short, so overview similarity is coarse.
- Ratings are approximate and static.
- No collaborative signals (popularity, watch history).

## 6. Literature Background
- **TF-IDF** (Term Frequency, Inverse Document Frequency) weights a term by how often it appears in a document and how rare it is across the collection, so distinctive attributes matter more than common ones (Salton & Buckley, 1988).
- **Cosine similarity** measures the angle between two vectors and is independent of their length, making it well suited to comparing sparse text-derived vectors.
- **Content-based recommenders** are surveyed in Ricci et al., *Recommender Systems Handbook*, and in Lops et al. (2011).

## 7. System Requirements

**Software**
| Item | Requirement |
|---|---|
| OS | Windows / macOS / Linux |
| Python | 3.8 or higher |
| Libraries | None (standard library only) |
| Tools | Git, any terminal or IDE |

**Hardware**: any computer able to run Python; the dataset and vectors use well under 10 MB of memory.

**Functional requirements**
- FR1: Load movie data from CSV.
- FR2: Recommend N similar movies for a title.
- FR3: Recommend from several liked titles.
- FR4: Filter by genre, year, rating, director and actor.
- FR5: Show top-rated movies.
- FR6: Tolerate misspelt titles.

**Non-functional requirements**: fast response (under 100 ms), modular code, readable output, no external dependencies.

## 8. System Design

### 8.1 Architecture
```
 movies.csv ──► Data Loader ──► Feature Extractor ──► TF-IDF Vectors
                                                            │
        User input (CLI) ──► Title Resolver (fuzzy) ────────┤
                                                            ▼
                                             Cosine Similarity Engine
                                                            │
                                     Ranker (score, rating tie-break)
                                                            │
                                                            ▼
                                                  Formatted Output
```

### 8.2 Dataset Design
`data/movies.csv` has 50 rows and 7 columns.

| Column | Description |
|---|---|
| title | Movie name |
| year | Release year |
| genres | Pipe-separated genres |
| director | One or more directors (pipe-separated) |
| cast | Lead actors (pipe-separated) |
| rating | Approximate audience rating (0–10) |
| overview | Short original plot summary |

### 8.3 Module Design
| Module | Responsibility |
|---|---|
| `Movie` dataclass | Holds metadata plus the computed vector and norm |
| `MovieRecommender._load` | Reads and validates CSV |
| `_raw_counts` / `_build_vectors` | Feature weighting and TF-IDF |
| `similarity` / `_cosine` | Similarity computation |
| `find_movie` | Exact, substring and fuzzy title lookup |
| `similar_to` | Single-title recommendations |
| `recommend_for_liked` | Multi-title profile recommendations |
| `search` / `top_rated` | Filtering |
| `main.py` | CLI menu and sub-commands |

## 9. Methodology and Algorithm

### 9.1 Feature weighting
Each movie yields a bag of terms with the following weights:

| Field | Weight | Example term |
|---|---|---|
| Genre | 3 | `genre_sci-fi` |
| Director | 2 | `dir_christopher_nolan` |
| Cast | 2 | `cast_leonardo_dicaprio` |
| Overview words | 1 | `wormhole` |

Names are stored as single tokens so "Christopher Nolan" is one feature rather than two unrelated words. Stop words are removed from overviews.

### 9.2 TF-IDF
For term *t* in movie *d* with weighted count *c*:

```
tf(t,d)  = 1 + ln(c)
idf(t)   = ln((1 + N) / (1 + df(t))) + 1
w(t,d)   = tf(t,d) × idf(t)
```
where *N* is the number of movies and *df(t)* the number of movies containing *t*.

### 9.3 Cosine similarity
```
sim(A, B) = Σ (A_t · B_t) / ( ||A|| · ||B|| )
```
Values range from 0 (nothing in common) to 1 (identical).

### 9.4 Algorithm: recommend similar movies
```
1. Resolve the user's title to a Movie (exact -> substring -> fuzzy)
2. For every other movie M: score = cosine(base, M)
3. Sort by score descending, then by rating descending
4. Return the top N
```

### 9.5 Algorithm: multi-movie profile
```
1. Resolve each liked title
2. profile = sum of (vector / norm) for each liked movie
3. For every movie not liked: score = cosine(profile, M)
4. Sort and return the top N
```

### 9.6 Complexity
With *N* movies and average *k* non-zero terms per vector, one recommendation costs O(N·k). Vector construction is a one-time O(N·k) cost at startup.

## 10. Implementation
- **Language:** Python 3.12 (works from 3.8).
- **Key modules used:** `csv`, `math`, `re`, `collections.Counter`, `difflib.get_close_matches`, `dataclasses`, `argparse`, `unittest`.
- **Design choices:**
  - Sparse dictionary vectors avoid heavy libraries like NumPy.
  - Vector norms are precomputed to speed up similarity.
  - Ties are broken by rating so better-rated movies surface first.
  - Errors raise `LookupError` / `ValueError` and are caught in the CLI to give friendly messages.

**Repository layout** is listed in `README.md`. Version control uses Git with the code published on GitHub.

## 11. Testing
Ten unit tests in `tests/test_recommender.py` were executed with `unittest`.

| # | Test | Purpose | Result |
|---|---|---|---|
| 1 | Dataset loaded | ≥ 50 rows read | Pass |
| 2 | Tokenizer | Stop words removed | Pass |
| 3 | Title lookup | Exact, case-insensitive and typo match | Pass |
| 4 | Similar excludes itself | Base movie not recommended, scores sorted | Pass |
| 5 | Sensible output | Inception → Nolan films appear | Pass |
| 6 | Self-similarity | Movie vs itself = 1.0 | Pass |
| 7 | Profile excludes liked | Liked movies never returned | Pass |
| 8 | Unknown title | Raises `LookupError` | Pass |
| 9 | Filters | Genre, rating, year, director respected | Pass |
| 10 | Top rated order | Descending by rating | Pass |

**Result: Ran 10 tests, OK.**

## 12. Results and Discussion

| Query | Top recommendations (match %) |
|---|---|
| Inception | The Prestige (13.3), The Dark Knight (13.2), Interstellar (12.6), Iron Man (12.1), Mad Max: Fury Road (12.0) |
| The Matrix | Avatar (14.4), Iron Man (11.8), Mad Max: Fury Road (11.7), Inception (11.0), Avengers: Endgame (10.6) |
| The Conjuring | Get Out (32.1), Kahaani (19.2), Parasite (16.1) |
| 3 Idiots + Dangal | PK (22.1), Lagaan (13.5), Taare Zameen Par (12.3), Gladiator (9.2), Zindagi Na Milegi Dobara (8.6) |

**Observations**
- Recommendations share genres, directors or actors with the query (for instance PK follows 3 Idiots and Dangal through Aamir Khan and Rajkumar Hirani).
- Horror queries return other Mystery/Thriller titles because genre tokens carry the highest weight.
- Absolute percentages are low because vectors contain many rare terms (names and plot words); the **ranking** is what matters, not the raw percentage.
- Results are instant for the current dataset.

## 13. Challenges Faced
- **Choosing weights:** too little genre weight produced random plot-word matches; too much made every action movie look identical. A 3:2:2:1 ratio gave balanced results.
- **Treating names as one token:** splitting "Christopher Nolan" into two words let unrelated people who share a first name match.
- **Typos in titles:** solved with difflib fuzzy matching.
- **Building without NumPy/scikit-learn:** implemented TF-IDF and cosine similarity manually with dictionaries.

## 14. Future Enhancements
1. Use larger public datasets such as MovieLens or TMDB.
2. Add collaborative filtering and build a hybrid recommender.
3. Add evaluation metrics: precision@k, recall@k, diversity.
4. Web interface using Flask or Streamlit, with posters.
5. Explanations such as "Because you liked X: same director and genre".
6. User profiles with persistent watch history.
7. Word embeddings or transformer sentence embeddings for richer plot similarity.

## 15. Conclusion
The project demonstrates a complete, explainable content-based recommender built from first principles. It converts movie metadata into TF-IDF vectors, ranks candidates with cosine similarity, supports multi-movie personalisation and rich filtering, and is validated by unit tests. The modular design and standard-library-only implementation make it easy to run, extend and publish on GitHub, and it provides a solid base for more advanced hybrid recommenders.

## 16. References
1. G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," *Information Processing & Management*, 24(5), 1988.
2. F. Ricci, L. Rokach, B. Shapira (eds.), *Recommender Systems Handbook*, Springer.
3. P. Lops, M. de Gemmis, G. Semeraro, "Content-based Recommender Systems: State of the Art and Trends," in *Recommender Systems Handbook*, 2011.
4. Python Software Foundation, Python 3 documentation: `csv`, `difflib`, `unittest`. https://docs.python.org/3/
5. F. M. Harper and J. A. Konstan, "The MovieLens Datasets: History and Context," *ACM TiiS*, 2015. (suggested for future work)
