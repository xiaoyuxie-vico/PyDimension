# Push Script Usage

## Quick Start

```bash
./scripts/push-all.sh
```

## Prerequisites

1. Be on `main` branch: `git checkout main`
2. Commit your changes: `git add . && git commit -m "message"`
3. Have both remotes configured: `git remote -v`

## What It Does

- Pushes to GitHub with clean README (no YAML)
- Pushes to Hugging Face Spaces with YAML frontmatter
- Restores clean README locally

## Example

```bash
git add streamlit_app.py
git commit -m "Update app"
./scripts/push-all.sh
```

That's it!

