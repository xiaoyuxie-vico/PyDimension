#!/bin/bash
# Setup convenient git aliases for PyDimension workflow

echo "🔧 Setting up git aliases..."

# Push to all remotes
git config --global alias.pushall '!f() { git push origin "${1:-main}" && git push space "${1:-main}"; }; f'

# Push to GitHub only
git config --global alias.pushgh '!f() { git push origin "${1:-main}"; }; f'

# Push to Hugging Face Spaces only
git config --global alias.pushhf '!f() { git push space "${1:-main}"; }; f'

# Quick status check
git config --global alias.statusall '!git remote -v && echo "" && git status'

echo "✅ Git aliases configured!"
echo ""
echo "Usage:"
echo "  git pushall          - Push to both GitHub and Hugging Face Spaces"
echo "  git pushall <branch> - Push specific branch to both remotes"
echo "  git pushgh           - Push to GitHub only"
echo "  git pushhf           - Push to Hugging Face Spaces only"
echo "  git statusall        - Show remotes and status"

