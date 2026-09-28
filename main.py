"""Command line interface for the Movie Recommendation System."""

import argparse
import sys

from src.recommender import MovieRecommender


def print_scored(results):
    if not results:
        print("No recommendations found.")
        return
    for i, (m, score) in enumerate(results, 1):
        print(f"{i:>2}. {m.summary()}  [match {score * 100:.1f}%]")


def print_movies(movies):
    if not movies:
        print("No movies matched your filters.")
        return
    for i, m in enumerate(movies, 1):
        print(f"{i:>2}. {m.summary()}")


def ask_int(prompt, default=None):
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print("Invalid number, ignoring.")
        return default


def ask_float(prompt, default=None):
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print("Invalid number, ignoring.")
        return default


def interactive(rec: MovieRecommender):
    menu = """
================ MOVIE RECOMMENDATION SYSTEM ================
 1. Recommend movies similar to a movie
 2. Recommend based on several movies you like
 3. Search / filter movies
 4. Top rated movies
 5. List all genres
 6. Show movie details
 0. Exit
==============================================================
"""
    while True:
        print(menu)
        choice = input("Enter choice: ").strip()
        try:
            if choice == "1":
                t = input("Movie you like: ")
                base = rec.find_movie(t)
                if base:
                    print(f"\nBecause you liked: {base.summary()}\n")
                print_scored(rec.similar_to(t, 5))
            elif choice == "2":
                raw = input("Enter liked movies separated by commas: ")
                titles = [x for x in raw.split(",") if x.strip()]
                print()
                print_scored(rec.recommend_for_liked(titles, 5))
            elif choice == "3":
                genre = input("Genre (blank = any): ").strip() or None
                yf = ask_int("From year (blank = any): ")
                yt = ask_int("To year (blank = any): ")
                mr = ask_float("Minimum rating (blank = any): ")
                d = input("Director contains (blank = any): ").strip() or None
                a = input("Actor contains (blank = any): ").strip() or None
                print()
                print_movies(rec.search(genre, yf, yt, mr, d, a, limit=15))
            elif choice == "4":
                genre = input("Genre (blank = all): ").strip() or None
                print()
                print_movies(rec.top_rated(10, genre))
            elif choice == "5":
                print(", ".join(rec.all_genres()))
            elif choice == "6":
                m = rec.find_movie(input("Movie title: "))
                if m is None:
                    print("Movie not found.")
                else:
                    print(f"\n{m.summary()}\nDirector: {', '.join(m.directors)}"
                          f"\nCast: {', '.join(m.cast)}\nPlot: {m.overview}")
            elif choice == "0":
                print("Thank you for using the system. Enjoy your movie!")
                break
            else:
                print("Invalid choice, try again.")
        except (LookupError, ValueError) as exc:
            print(f"Error: {exc}")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break


def build_parser():
    p = argparse.ArgumentParser(description="Movie Recommendation System")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("similar", help="movies similar to a title")
    s.add_argument("title")
    s.add_argument("-n", type=int, default=5)
    l = sub.add_parser("liked", help="recommend from several liked titles")
    l.add_argument("titles", nargs="+")
    l.add_argument("-n", type=int, default=5)
    t = sub.add_parser("top", help="top rated movies")
    t.add_argument("-g", "--genre")
    t.add_argument("-n", type=int, default=10)
    f = sub.add_parser("search", help="filter movies")
    f.add_argument("--genre")
    f.add_argument("--from-year", type=int)
    f.add_argument("--to-year", type=int)
    f.add_argument("--min-rating", type=float)
    f.add_argument("--director")
    f.add_argument("--actor")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    rec = MovieRecommender()
    try:
        if args.cmd is None:
            interactive(rec)
        elif args.cmd == "similar":
            print_scored(rec.similar_to(args.title, args.n))
        elif args.cmd == "liked":
            print_scored(rec.recommend_for_liked(args.titles, args.n))
        elif args.cmd == "top":
            print_movies(rec.top_rated(args.n, args.genre))
        elif args.cmd == "search":
            print_movies(rec.search(args.genre, args.from_year, args.to_year,
                                    args.min_rating, args.director, args.actor, 15))
    except (LookupError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
