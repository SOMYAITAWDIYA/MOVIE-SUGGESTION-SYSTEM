# 🎬 Movie Recommendation System

A content-based movie recommendation engine written in pure Python (standard library only). It suggests movies similar to the ones you like using **TF-IDF vectors** and **cosine similarity** over genres, directors, cast and plot descriptions.

> **VITyarthi Project** | Course: `<Problem Solving and Programming & CSE1021>` | Student: `<SOMYA ITAWDIYA>` | Reg. No: `<26BAI10424>`

---

## ✨ Features

- **Similar-movie recommendations**: type one movie, get the top 5 closest matches with a match percentage.
- **Multi-movie taste profile**: give several liked movies and get personalised suggestions.
- **Advanced search filters**: genre, year range, minimum rating, director, actor.
- **Top-rated lists**, overall or by genre.
- **Typo tolerant**: "Intersteller" still finds *Interstellar*.
- **Two ways to use it**: interactive menu or one-line CLI commands.
- **Zero dependencies** and a unit test suite (10 tests).

## 📁 Project Structure

```
movie-recommendation-system/
├── main.py                  # CLI (menu + sub-commands)
├── src/
│   ├── __init__.py
│   └── recommender.py       # TF-IDF + cosine similarity engine
├── data/
│   └── movies.csv           # 50-movie dataset (Hollywood + Indian cinema)
├── tests/
│   ├── __init__.py
│   └── test_recommender.py  # unit tests
├── docs/
│   ├── PROJECT_DESCRIPTION.md
│   └── PROJECT_REPORT.md    # full academic report
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## 🚀 Getting Started

**Requirements:** Python 3.8 or higher. No `pip install` needed.

```bash
git clone https://github.com/<your-username>/movie-recommendation-system.git
cd movie-recommendation-system
python main.py            # Windows
python3 main.py           # macOS / Linux
```

## 🖥️ Usage

### Interactive menu
```bash
python3 main.py
```

### Direct commands
```bash
# Movies similar to a title
python3 main.py similar "Inception" -n 5

# Recommendations from several liked movies
python3 main.py liked "3 Idiots" "Dangal"

# Top 5 comedies
python3 main.py top -g Comedy -n 5

# Filtered search
python3 main.py search --genre Horror --min-rating 7.5 --from-year 2010
python3 main.py search --director Nolan
```

### Sample output
```
$ python3 main.py similar "The Conjuring" -n 3
 1. Get Out (2017)  | Horror, Mystery, Thriller | Rating: 7.7  [match 32.1%]
 2. Kahaani (2012)  | Mystery, Thriller         | Rating: 8.1  [match 19.2%]
 3. Parasite (2019) | Comedy, Drama, Thriller   | Rating: 8.5  [match 16.1%]
```

## 🧠 How It Works

1. **Feature extraction**: each movie is converted into weighted terms:
   genres ×3, director ×2, cast ×2, plot words ×1 (stop words removed).
2. **TF-IDF weighting**: `tf = 1 + ln(count)`, `idf = ln((1+N)/(1+df)) + 1`. Rare terms (e.g. a specific director) count more than common ones.
3. **Cosine similarity**: `sim(A,B) = (A · B) / (‖A‖ ‖B‖)`.
4. **Ranking**: movies are sorted by similarity, with rating as the tie-breaker.
5. **User profile**: for multiple liked movies, their normalised vectors are summed into one profile vector and compared with every unseen movie.

## ✅ Running the Tests

```bash
python3 -m unittest discover -s tests -t . -v
```

## ➕ Adding More Movies

Append rows to `data/movies.csv` using the same columns:
`title,year,genres,director,cast,rating,overview`. Use `|` to separate multiple genres, directors or cast members and wrap the overview in double quotes. The engine rebuilds all vectors automatically at startup.

## 🔮 Future Enhancements

- Load the full MovieLens / TMDB dataset
- Hybrid model with collaborative filtering
- Web UI (Flask / Streamlit)
- User accounts and rating history
- Evaluation metrics (precision@k, recall@k)

## 📄 Documentation

- [Project Description](docs/PROJECT_DESCRIPTION.md)
- [Full Project Report](docs/PROJECT_REPORT.md)

## 📜 License

Released under the [MIT License](LICENSE).

> Ratings in the dataset are approximate and included for educational purposes only. Plot summaries are original short descriptions.
