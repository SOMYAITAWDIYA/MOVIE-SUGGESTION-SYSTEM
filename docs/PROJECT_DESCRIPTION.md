# Project Description

**Title:** Movie Recommendation System using Content-Based Filtering
**Platform:** VITyarthi
**Language:** Python 3 (standard library only)
**Student:** `<YOUR NAME>` | **Reg. No:** `<YOUR REG NO>`

## Overview
Choosing what to watch among thousands of titles is time-consuming. This project builds an intelligent recommender that studies the content of movies (genre, director, cast and story) and suggests films similar to those a user already enjoys.

## Problem Statement
Users face information overload on streaming platforms. A lightweight, explainable system is needed that can recommend relevant movies without requiring a large user-rating history (the "cold start" problem).

## Objectives
1. Represent every movie as a numerical feature vector.
2. Compute similarity between movies using cosine similarity.
3. Recommend the top-N most similar movies for one or many liked titles.
4. Provide filters (genre, year, rating, director, actor) and a friendly CLI.
5. Keep the code modular, tested and easy to publish on GitHub.

## Technologies Used
| Area | Tool |
|---|---|
| Language | Python 3.8+ |
| Algorithms | TF-IDF, Cosine Similarity, Fuzzy string matching (difflib) |
| Data | CSV file (50 movies) |
| Testing | unittest |
| Version control | Git & GitHub |

## Modules
- **Data loader**: reads and validates `movies.csv`.
- **Vectoriser**: builds weighted TF-IDF vectors.
- **Recommender**: similar-movie and taste-profile recommendations.
- **Filter engine**: multi-criteria search and top-rated lists.
- **CLI**: interactive menu and sub-commands.

## Expected Outcome
A working command-line application that returns relevant, ranked movie recommendations with match percentages, backed by unit tests and full documentation.
