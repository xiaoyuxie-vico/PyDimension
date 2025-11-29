# Development Workflow Guide

This guide covers convenient workflows for working with PyDimension, including easy deployment to GitHub and Hugging Face Spaces.

## Quick Push Workflow

### Option 1: Use the Push Script (Recommended)

We provide a convenient script that pushes to both remotes:

```bash
# Make sure you're on main branch and have committed your changes
./scripts/push-all.sh
```

The script will:
- ✅ Check you're on the correct branch
- ✅ Warn about uncommitted changes
- ✅ Push to GitHub (origin)
- ✅ Push to Hugging Face Spaces (space)
- ✅ Show success/error messages

### Option 2: Use Git Aliases

Set up convenient git aliases:

```bash
# Run once to set up aliases
./scripts/setup-git-aliases.sh
```

Then use:

```bash
# Push to both remotes
git pushall

# Push to GitHub only
git pushgh

# Push to Hugging Face Spaces only
git pushhf

# Check status and remotes
git statusall
```

### Option 3: Manual Push

```bash
# Push to both remotes manually
git push origin main
git push space main
```

## Standard Workflow

### 1. Make Changes

```bash
# Edit files
vim streamlit_app.py
# or use your favorite editor
```

### 2. Stage and Commit

```bash
# Stage all changes
git add .

# Commit with a descriptive message
git commit -m "Add new feature: description"
```

### 3. Push to Remotes

**Using the script:**
```bash
./scripts/push-all.sh
```

**Using git aliases:**
```bash
git pushall
```

**Manual:**
```bash
git push origin main
git push space main
```

## Branch Management

### Working on a Feature Branch

```bash
# Create and switch to new branch
git checkout -b feature/new-feature

# Make changes and commit
git add .
git commit -m "Add new feature"

# Push to GitHub (for review/backup)
git push origin feature/new-feature

# When ready, merge to main
git checkout main
git merge feature/new-feature

# Push to both remotes
./scripts/push-all.sh
```

### Updating Main Branch

```bash
# Pull latest from GitHub
git pull origin main

# Make your changes
# ...

# Commit and push
git add .
git commit -m "Update: description"
./scripts/push-all.sh
```

## Hugging Face Spaces Deployment

### First-Time Setup

1. **Create Space on Hugging Face:**
   - Go to https://huggingface.co/spaces
   - Click "New Space"
   - Choose Streamlit SDK
   - Name it (e.g., `PyDimension`)

2. **Add Remote:**
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   ```

3. **Verify:**
   ```bash
   git remote -v
   ```

### Regular Deployment

After making changes:

```bash
# Commit changes
git add .
git commit -m "Your commit message"

# Push to both remotes
./scripts/push-all.sh
```

The Hugging Face Space will automatically rebuild when you push.

### Force Push (if needed)

⚠️ **Use with caution!**

```bash
# Only if you've cleaned history or rebased
git push space main --force
```

## Troubleshooting

### Remote Not Found

**Error:** `fatal: 'space' does not appear to be a git repository`

**Solution:**
```bash
git remote add space https://huggingface.co/spaces/xiaoyuxie-vico/PyDimension
```

### Push Rejected (Binary Files)

**Error:** `Your push was rejected because it contains binary files`

**Solution:**
1. Remove binary files from history (see [HUGGINGFACE_DEPLOY.md](HUGGINGFACE_DEPLOY.md))
2. Or use external URLs for images
3. Ensure `.gitignore` excludes binary files

### Uncommitted Changes

**Error:** Script warns about uncommitted changes

**Solution:**
```bash
# Option 1: Commit them
git add .
git commit -m "Your message"

# Option 2: Stash them
git stash
# ... do your push ...
git stash pop
```

### Wrong Branch

**Error:** Script warns you're not on main

**Solution:**
```bash
# Switch to main
git checkout main

# Or continue anyway (if you know what you're doing)
```

## Advanced Workflows

### Pre-Push Hook (Optional)

Create `.git/hooks/pre-push` to automatically run checks:

```bash
#!/bin/bash
# Run tests before pushing
python -m pytest tests/ || exit 1
```

### Automated Deployment

For CI/CD, see GitHub Actions examples in `.github/workflows/` (if you create them).

### Multiple Environments

If you have multiple Spaces (dev, staging, prod):

```bash
git remote add space-dev https://huggingface.co/spaces/user/pydimension-dev
git remote add space-prod https://huggingface.co/spaces/user/pydimension-prod

# Push to specific environment
git push space-dev main
git push space-prod main
```

## Best Practices

1. **Always commit before pushing** - Don't push uncommitted changes
2. **Use descriptive commit messages** - Helps track changes
3. **Test locally first** - Run `streamlit run streamlit_app.py` before deploying
4. **Check build logs** - Monitor Hugging Face Space logs after pushing
5. **Keep main branch clean** - Use feature branches for development
6. **Regular backups** - Push to GitHub regularly, not just before deployment

## Quick Reference

```bash
# Setup (one-time)
./scripts/setup-git-aliases.sh

# Daily workflow
git add .
git commit -m "Description"
./scripts/push-all.sh

# Or with aliases
git add .
git commit -m "Description"
git pushall
```

## Related Documentation

- [HUGGINGFACE_DEPLOY.md](HUGGINGFACE_DEPLOY.md) - Detailed Hugging Face deployment guide
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development workflow and code checking
- [SETUP.md](SETUP.md) - Installation and setup

---

**Tip:** Add the scripts directory to your PATH or create a symlink for even easier access:
```bash
ln -s $(pwd)/scripts/push-all.sh ~/bin/push-pydimension
```

