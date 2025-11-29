# Deploying PyDimension to Hugging Face Spaces

This guide covers how to deploy the PyDimension Streamlit app to Hugging Face Spaces.

## Quick Start

If you've already set up the deployment, simply push to the main branch:

```bash
git push space main
```

Or if you need to set up the remote:

```bash
git remote add space https://huggingface.co/spaces/xiaoyuxie-vico/PyDimension
git push space main
```

## Prerequisites

1. **Hugging Face Account**: Sign up at [huggingface.co](https://huggingface.co)
2. **Hugging Face Space Created**: Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
   - Choose **Streamlit** as the SDK
   - Name it (e.g., `PyDimension`)
3. **Git Repository**: Your code should be in a git repository

## Initial Setup

### Step 1: Create Hugging Face Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"New Space"**
3. Fill in:
   - **Space name**: `PyDimension` (or your preferred name)
   - **SDK**: Select **Streamlit**
   - **Visibility**: Public (or Private)
4. Click **"Create Space"**

### Step 2: Add Hugging Face Remote

Add the Hugging Face Space as a git remote:

```bash
cd /Users/xie/projects/PyDimension
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

Replace `YOUR_USERNAME` and `YOUR_SPACE_NAME` with your actual values.

### Step 3: Configure README.md

The README.md must include YAML frontmatter for Hugging Face Spaces configuration:

```yaml
---
title: PyDimension
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.28.0"
app_file: streamlit_app.py
pinned: false
---
```

**Configuration Options:**
- `title`: Display name for your Space
- `emoji`: Icon/emoji for your Space
- `colorFrom` / `colorTo`: Gradient colors for Space header
- `sdk`: Must be `streamlit` for Streamlit apps
- `sdk_version`: Streamlit version (should match `requirements.txt`)
- `app_file`: Main Streamlit app file (usually `streamlit_app.py`)
- `pinned`: Whether to pin the Space to your profile

### Step 4: Ensure Required Files

Your repository must include:

1. **`streamlit_app.py`** - Main Streamlit application (in root directory)
2. **`requirements.txt`** - Python dependencies
3. **`README.md`** - With YAML frontmatter (see Step 3)
4. **`.streamlit/config.toml`** (optional) - Streamlit configuration

### Step 5: Handle Binary Files

Hugging Face Spaces **rejects binary files** (images, large files) in git history.

**Solution 1: Remove from History (Recommended)**

```bash
# Remove images directory from git history
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --force \
  --index-filter 'git rm --cached --ignore-unmatch -r images/' \
  --prune-empty --tag-name-filter cat -- --all

# Push cleaned branch
git push space main --force
```

**Solution 2: Use External URLs**

Instead of storing images locally, use external URLs:
- Imgur: https://imgur.com
- PostImg: https://postimg.cc
- GitHub Releases

Then reference them in code:
```python
st.image("https://i.postimg.cc/Gt8Cznt4/logo.png")
```

**Solution 3: Update .gitignore**

Ensure `.gitignore` excludes binary files:

```gitignore
# Images directory (for Hugging Face Spaces compatibility)
images/
*.png
```

## Deployment

### First Deployment

```bash
# Ensure you're on main branch
git checkout main

# Push to Hugging Face Spaces
git push space main
```

### Subsequent Updates

After making changes:

```bash
# Commit your changes
git add .
git commit -m "Your commit message"

# Push to Hugging Face Spaces
git push space main
```

The Space will automatically rebuild when you push.

## File Structure

Your repository structure should look like:

```
PyDimension/
├── streamlit_app.py          # Main Streamlit app (REQUIRED)
├── requirements.txt          # Python dependencies (REQUIRED)
├── README.md                 # With YAML frontmatter (REQUIRED)
├── .streamlit/
│   └── config.toml          # Streamlit config (OPTIONAL)
├── pydimension/             # Package code
│   ├── __init__.py
│   ├── data_generation/
│   ├── data_preprocessing/
│   ├── dimensional_analysis/
│   ├── constraint_filtering/
│   ├── optimization_discovery/
│   └── configs/
└── ... (other files)
```

## Common Issues and Solutions

### Issue 1: Binary Files Rejected

**Error:**
```
remote: Your push was rejected because it contains binary files.
remote: Offending files:
remote:   - images/PDE.png
```

**Solution:**
1. Remove binary files from git history (see Step 5 above)
2. Or use external URLs for images
3. Ensure `.gitignore` excludes binary files

### Issue 2: Configuration Error

**Error:**
```
Missing configuration in README
```

**Solution:**
Add YAML frontmatter to README.md (see Step 3 above).

### Issue 3: Streamlit Compatibility

**Error:**
```
TypeError: ImageMixin.image() got an unexpected keyword argument 'use_container_width'
```

**Solution:**
Remove `use_container_width=True` from `st.image()` calls. Use:
```python
st.image("path/to/image.png")  # Instead of st.image(..., use_container_width=True)
```

### Issue 4: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'pydimension'
```

**Solution:**
Ensure `requirements.txt` includes all dependencies and the package structure is correct. The package should be importable as:
```python
from pydimension import ...
```

### Issue 5: Build Timeout

**Error:**
Build takes too long or times out.

**Solution:**
- Reduce dependencies in `requirements.txt`
- Use lighter versions of packages
- Check build logs in the Space's "Logs" tab

## Monitoring Deployment

### Check Build Status

1. Go to your Space page: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`
2. Click **"Logs"** tab to see build progress
3. Wait 2-5 minutes for initial build

### View Logs

The **Logs** tab shows:
- Installation progress
- Build errors
- Runtime errors
- Application output

### Test Your App

Once built, your app will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

## Best Practices

### 1. Keep Dependencies Minimal

Only include necessary packages in `requirements.txt`:
```txt
numpy>=1.21.0
pandas>=1.3.0
streamlit>=1.28.0
# ... only what you need
```

### 2. Use Specific Versions

Pin versions for reproducibility:
```txt
streamlit==1.28.0
numpy==1.24.0
```

### 3. Test Locally First

Always test your app locally before deploying:
```bash
streamlit run streamlit_app.py
```

### 4. Handle Large Files

- Use external storage for large datasets
- Use Hugging Face Datasets for data files
- Keep repository size small

### 5. Environment Variables

For sensitive data, use Hugging Face Secrets:
1. Go to Space Settings → Secrets
2. Add environment variables
3. Access in code: `os.getenv("YOUR_SECRET")`

## Updating Your Space

### Regular Updates

```bash
# Make changes to your code
# ...

# Commit and push
git add .
git commit -m "Update: description of changes"
git push space main
```

### Force Update (if needed)

```bash
git push space main --force
```

⚠️ **Warning**: Only use `--force` if necessary, as it overwrites remote history.

## Troubleshooting

### App Won't Start

1. Check **Logs** tab for errors
2. Verify `streamlit_app.py` is in root directory
3. Ensure `app_file` in README.md matches your file name
4. Check that all imports work

### Dependencies Not Installing

1. Verify `requirements.txt` syntax
2. Check for version conflicts
3. Ensure all packages are available on PyPI
4. Review build logs for specific errors

### Import Errors

1. Ensure package structure is correct
2. Check that `pydimension/__init__.py` exists
3. Verify all modules are included in repository
4. Test imports locally first

## Advanced Configuration

### Custom Streamlit Config

Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Multiple Spaces

If you want to deploy to multiple Spaces:

```bash
# Add multiple remotes
git remote add space-prod https://huggingface.co/spaces/user/prod
git remote add space-dev https://huggingface.co/spaces/user/dev

# Push to specific Space
git push space-prod main
git push space-dev main
```

## Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Spaces Configuration Reference](https://huggingface.co/docs/hub/spaces-config-reference)
- [PyDimension GitHub Repository](https://github.com/xiaoyuxie-vico/PyDimension)

## Support

If you encounter issues:

1. Check the **Logs** tab in your Space
2. Review this documentation
3. Check [Hugging Face Spaces Discussions](https://huggingface.co/spaces)
4. Open an issue on the [PyDimension GitHub repository](https://github.com/xiaoyuxie-vico/PyDimension/issues)

---

**Last Updated**: November 2024

