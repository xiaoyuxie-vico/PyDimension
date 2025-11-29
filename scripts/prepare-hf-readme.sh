#!/bin/bash
# Prepare README.md for Hugging Face Spaces by adding YAML frontmatter
# This allows keeping a clean README on GitHub while having YAML for HF Spaces

HF_YAML="---
title: PyDimension
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: \"1.28.0\"
app_file: streamlit_app.py
pinned: false
---"

# Check if README.md exists
if [ ! -f "README.md" ]; then
    echo "Error: README.md not found"
    exit 1
fi

# Check if YAML frontmatter already exists
if head -n 1 README.md | grep -q "^---$"; then
    echo "✅ README.md already has YAML frontmatter"
    exit 0
fi

# Create temporary backup for processing
TEMP_BACKUP="README.md.backup"
cp README.md "$TEMP_BACKUP"

# Prepend YAML to README
echo "$HF_YAML" > README.md.tmp
echo "" >> README.md.tmp
cat "$TEMP_BACKUP" >> README.md.tmp
mv README.md.tmp README.md

# Clean up temporary backup
rm -f "$TEMP_BACKUP"

echo "✅ Added YAML frontmatter to README.md for Hugging Face Spaces"

