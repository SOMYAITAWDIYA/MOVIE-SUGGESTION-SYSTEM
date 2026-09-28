#!/usr/bin/env bash
# Usage: ./upload_to_github.sh https://github.com/<username>/movie-recommendation-system.git
set -e
if [ -z "$1" ]; then echo "Usage: $0 <github-repo-url>"; exit 1; fi
git init
git add .
git commit -m "Initial commit: Movie Recommendation System"
git branch -M main
git remote add origin "$1"
git push -u origin main
