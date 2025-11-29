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
    
    # Fetch latest from Hugging Face Space to check if we're behind
    echo "📥 Fetching latest from Hugging Face Spaces..."
    git fetch space "$CURRENT_BRANCH" 2>/dev/null || true
    
    # Always create commit directly on remote HEAD to avoid binary file history issues
    # The Hugging Face Space has a clean history (no binary files), so we build on top of it
    # This avoids bringing in the local history which may contain binary file references
    REMOTE_HEAD=$(git rev-parse "space/$CURRENT_BRANCH" 2>/dev/null || echo "")
    
    if [ -z "$REMOTE_HEAD" ]; then
        echo "⚠️  Could not determine remote HEAD. Attempting direct push..."
        # Fallback: ensure README has YAML and try direct push
        if ! head -n 1 README.md | grep -q "^---$"; then
            ./scripts/prepare-hf-readme.sh
            git add README.md
            git commit -m "Add YAML frontmatter for Hugging Face Spaces configuration" || true
        fi
        if git push space "$CURRENT_BRANCH" 2>&1; then
            echo "✅ Successfully pushed to Hugging Face Spaces"
        else
            echo "❌ Failed to push to Hugging Face Spaces"
            exit 1
        fi
    else
        echo "📝 Creating commit directly on remote HEAD (clean history) to avoid binary file issues..."
        
        # Save current branch name
        CURRENT_BRANCH_NAME=$(git branch --show-current)
        
        # Create temporary branch from clean remote HEAD
        TEMP_BRANCH="temp-hf-push-$$"
        git checkout -b "$TEMP_BRANCH" "$REMOTE_HEAD" 2>/dev/null || {
            # If branch exists, delete and recreate
            git branch -D "$TEMP_BRANCH" 2>/dev/null || true
            git checkout -b "$TEMP_BRANCH" "$REMOTE_HEAD" 2>/dev/null || exit 1
        }
        
        # Get README with YAML from the original branch (before we switched)
        ORIGINAL_README=$(git show "${CURRENT_BRANCH_NAME}:README.md" 2>/dev/null || echo "")
        
        if [ -z "$ORIGINAL_README" ] || ! echo "$ORIGINAL_README" | head -n 1 | grep -q "^---$"; then
            # README doesn't have YAML, add it
            echo "$ORIGINAL_README" > README.md 2>/dev/null || true
            if [ ! -s README.md ]; then
                # File is empty or doesn't exist, get it from HEAD
                git checkout "${CURRENT_BRANCH_NAME}" -- README.md 2>/dev/null || true
            fi
            ./scripts/prepare-hf-readme.sh
        else
            # README already has YAML, use it
            echo "$ORIGINAL_README" > README.md
        fi
        
        # Verify YAML is present
        if ! head -n 1 README.md | grep -q "^---$"; then
            echo "❌ Error: Failed to ensure README has YAML frontmatter"
            git checkout "$CURRENT_BRANCH_NAME" 2>/dev/null || true
            git branch -D "$TEMP_BRANCH" 2>/dev/null || true
            exit 1
        fi
        
        # Commit the README with YAML
        git add README.md
        git commit -m "Add YAML frontmatter for Hugging Face Spaces configuration" || true
        
        # Push from temp branch to main on remote
        if git push space "$TEMP_BRANCH:$CURRENT_BRANCH"; then
            echo "✅ Successfully pushed to Hugging Face Spaces"
            # Switch back to original branch
            git checkout "$CURRENT_BRANCH_NAME" 2>/dev/null || true
            git branch -D "$TEMP_BRANCH" 2>/dev/null || true
        else
            echo "❌ Failed to push to Hugging Face Spaces"
            git checkout "$CURRENT_BRANCH_NAME" 2>/dev/null || true
            git branch -D "$TEMP_BRANCH" 2>/dev/null || true
            exit 1
        fi
    fi
else
    echo "⚠️  No 'space' remote found, skipping Hugging Face Spaces"
    echo "   Add it with: git remote add space https://huggingface.co/spaces/xiaoyuxie-vico/PyDimension"
fi

# Restore clean README locally and ensure GitHub has it too
echo ""
echo "📝 Ensuring clean README.md for GitHub (removing YAML frontmatter)..."
./scripts/restore-github-readme.sh
rm -f README.md.backup

# Verify README is clean (no YAML)
if head -n 1 README.md | grep -q "^---$"; then
    echo "⚠️  Warning: README.md still has YAML frontmatter after restore attempt"
    # Force remove YAML
    YAML_END=$(grep -n "^---$" README.md | sed -n '2p' | cut -d: -f1)
    if [ -n "$YAML_END" ]; then
        sed "1,$((YAML_END+1))d" README.md > README.md.tmp
        mv README.md.tmp README.md
        echo "✅ Force-removed YAML frontmatter"
    fi
fi

# Commit and push clean README to GitHub if it changed
if ! git diff --quiet README.md 2>/dev/null; then
    echo "📝 Committing clean README.md (no YAML) to GitHub..."
    git add README.md
    git commit -m "Keep README clean for GitHub (remove YAML frontmatter)" || true
    
    if git remote get-url origin >/dev/null 2>&1; then
        echo "📦 Pushing clean README to GitHub..."
        if git push origin "$CURRENT_BRANCH"; then
            echo "✅ Successfully pushed clean README to GitHub"
        else
            echo "⚠️  Warning: Failed to push clean README to GitHub (but local is clean)"
        fi
    fi
fi

# Clean up backup file
rm -f "$README_BACKUP"

echo ""
echo "✅ Local README.md is now clean (no YAML frontmatter)"
echo ""
echo "🎉 All done! Your changes are now live on both platforms."
echo "   - GitHub: Clean README (no YAML) ✅"
echo "   - Hugging Face Spaces: README with YAML frontmatter for configuration ✅"

# Restore clean README locally and ensure GitHub has it too
echo ""
echo "📝 Ensuring clean README.md for GitHub (removing YAML frontmatter)..."
./scripts/restore-github-readme.sh
rm -f README.md.backup

# Verify README is clean (no YAML)
if head -n 1 README.md | grep -q "^---$"; then
    echo "⚠️  Warning: README.md still has YAML frontmatter after restore attempt"
    # Force remove YAML
    YAML_END=$(grep -n "^---$" README.md | sed -n '2p' | cut -d: -f1)
    if [ -n "$YAML_END" ]; then
        sed "1,$((YAML_END+1))d" README.md > README.md.tmp
        mv README.md.tmp README.md
        echo "✅ Force-removed YAML frontmatter"
    fi
fi

# Commit and push clean README to GitHub if it changed
if ! git diff --quiet README.md 2>/dev/null; then
    echo "📝 Committing clean README.md (no YAML) to GitHub..."
    git add README.md
    git commit -m "Keep README clean for GitHub (remove YAML frontmatter)" || true
    
    if git remote get-url origin >/dev/null 2>&1; then
        echo "📦 Pushing clean README to GitHub..."
        if git push origin "$CURRENT_BRANCH"; then
            echo "✅ Successfully pushed clean README to GitHub"
        else
            echo "⚠️  Warning: Failed to push clean README to GitHub (but local is clean)"
        fi
    fi
fi

# Clean up backup file
rm -f "$README_BACKUP"

echo ""
echo "✅ Local README.md is now clean (no YAML frontmatter)"
echo ""
echo "🎉 All done! Your changes are now live on both platforms."
echo "   - GitHub: Clean README (no YAML) ✅"
echo "   - Hugging Face Spaces: README with YAML frontmatter for configuration ✅"

