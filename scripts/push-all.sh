#!/bin/bash
# Push to both GitHub and Hugging Face Spaces remotes
# Automatically manages YAML frontmatter: clean for GitHub, with YAML for HF Spaces

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

# Check if README has YAML frontmatter
README_HAD_YAML=false
if head -n 1 README.md | grep -q "^---$"; then
    README_HAD_YAML=true
    echo "✅ README.md already has YAML frontmatter"
else
    echo "📝 README.md is clean (no YAML frontmatter)"
fi

# Save current README state
README_BACKUP="README.md.push-backup"
cp README.md "$README_BACKUP"

# Ensure README is clean for GitHub push
if [ "$README_HAD_YAML" = true ]; then
    echo "📝 Removing YAML frontmatter for GitHub push..."
    ./scripts/restore-github-readme.sh
    rm -f README.md.backup
fi

# Push to GitHub first (with clean README, no YAML)
if git remote get-url origin >/dev/null 2>&1; then
    echo "📦 Pushing to GitHub (origin) with clean README..."
    if git push origin "$CURRENT_BRANCH"; then
        echo "✅ Successfully pushed to GitHub"
    else
        echo "❌ Failed to push to GitHub"
        # Restore original README state
        mv "$README_BACKUP" README.md 2>/dev/null || true
        exit 1
    fi
else
    echo "⚠️  No 'origin' remote found, skipping GitHub"
fi

# Restore README with YAML for Hugging Face Spaces
if [ "$README_HAD_YAML" = true ]; then
    # Restore from backup
    mv "$README_BACKUP" README.md
    echo "✅ Restored README.md with YAML frontmatter"
elif [ -f "$README_BACKUP" ]; then
    # If it didn't have YAML, restore from backup and add YAML
    mv "$README_BACKUP" README.md
fi

# Ensure YAML frontmatter is present for Hugging Face Spaces
if ! head -n 1 README.md | grep -q "^---$"; then
    echo "📝 Adding YAML frontmatter to README.md for Hugging Face Spaces..."
    ./scripts/prepare-hf-readme.sh
fi

# Verify YAML frontmatter is correct and complete
if ! head -n 1 README.md | grep -q "^---$"; then
    echo "❌ Error: Failed to add YAML frontmatter to README.md"
    exit 1
fi

# Verify required YAML fields are present
REQUIRED_FIELDS=("title" "sdk" "app_file")
MISSING_FIELDS=()
for field in "${REQUIRED_FIELDS[@]}"; do
    if ! grep -q "^${field}:" README.md; then
        MISSING_FIELDS+=("$field")
    fi
done

if [ ${#MISSING_FIELDS[@]} -gt 0 ]; then
    echo "❌ Error: YAML frontmatter is missing required fields: ${MISSING_FIELDS[*]}"
    exit 1
fi

echo "✅ README.md now has complete YAML frontmatter for Hugging Face Spaces"

# Push to Hugging Face Spaces (space) with YAML
if git remote get-url space >/dev/null 2>&1; then
    echo "🌐 Pushing to Hugging Face Spaces (space) with YAML frontmatter..."
    
    # Create a temporary commit with YAML if needed
    TEMP_COMMIT=false
    if ! git diff --quiet README.md 2>/dev/null; then
        echo "📝 Staging README.md with YAML frontmatter..."
        git add README.md
        git commit -m "Add YAML frontmatter for Hugging Face Spaces configuration" || true
        TEMP_COMMIT=true
    fi
    
    if git push space "$CURRENT_BRANCH"; then
        echo "✅ Successfully pushed to Hugging Face Spaces"
        
        # If we created a temp commit, we can optionally keep it or reset
        # For now, we'll keep it since it's needed for HF Spaces
        if [ "$TEMP_COMMIT" = true ]; then
            echo "ℹ️  Note: A commit with YAML frontmatter was created for Hugging Face Spaces"
        fi
    else
        echo "❌ Failed to push to Hugging Face Spaces"
        # Reset temp commit if it was created
        if [ "$TEMP_COMMIT" = true ]; then
            git reset HEAD~1 2>/dev/null || true
        fi
        exit 1
    fi
else
    echo "⚠️  No 'space' remote found, skipping Hugging Face Spaces"
    echo "   Add it with: git remote add space https://huggingface.co/spaces/xiaoyuxie-vico/PyDimension"
fi

# Restore clean README locally (remove YAML if we added it)
if [ "$README_HAD_YAML" = false ]; then
    echo ""
    echo "📝 Restoring clean README.md locally (removing YAML frontmatter)..."
    ./scripts/restore-github-readme.sh
    rm -f README.md.backup
    echo "✅ Local README.md is now clean (no YAML frontmatter)"
else
    # Clean up backup file
    rm -f "$README_BACKUP"
fi

echo ""
echo "🎉 All done! Your changes are now live on both platforms."
echo "   - GitHub: Clean README (no YAML)"
echo "   - Hugging Face Spaces: README with YAML frontmatter for configuration"

