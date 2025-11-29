#!/bin/bash
# Push to both GitHub and Hugging Face Spaces remotes

set -e  # Exit on error

echo "🚀 Pushing to all remotes..."

# Check if we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "⚠️  Warning: You're on branch '$CURRENT_BRANCH', not 'main'"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if there are uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  Warning: You have uncommitted changes"
    read -p "Commit them first? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add -A
        read -p "Enter commit message: " COMMIT_MSG
        git commit -m "$COMMIT_MSG"
    else
        echo "❌ Please commit or stash your changes first"
        exit 1
    fi
fi

# Push to GitHub (origin)
if git remote get-url origin >/dev/null 2>&1; then
    echo "📦 Pushing to GitHub (origin)..."
    if git push origin "$CURRENT_BRANCH"; then
        echo "✅ Successfully pushed to GitHub"
    else
        echo "❌ Failed to push to GitHub"
        exit 1
    fi
else
    echo "⚠️  No 'origin' remote found, skipping GitHub"
fi

# Push to Hugging Face Spaces (space)
if git remote get-url space >/dev/null 2>&1; then
    echo "🌐 Pushing to Hugging Face Spaces (space)..."
    if git push space "$CURRENT_BRANCH"; then
        echo "✅ Successfully pushed to Hugging Face Spaces"
    else
        echo "❌ Failed to push to Hugging Face Spaces"
        exit 1
    fi
else
    echo "⚠️  No 'space' remote found, skipping Hugging Face Spaces"
    echo "   Add it with: git remote add space https://huggingface.co/spaces/xiaoyuxie-vico/PyDimension"
fi

echo ""
echo "🎉 All done! Your changes are now live on both platforms."

