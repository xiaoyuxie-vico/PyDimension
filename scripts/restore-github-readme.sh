#!/bin/bash
# Restore clean README.md for GitHub (remove YAML frontmatter)

if [ ! -f "README.md" ]; then
    echo "Error: README.md not found"
    exit 1
fi

# Check if YAML frontmatter exists
if ! head -n 1 README.md | grep -q "^---$"; then
    echo "✅ README.md doesn't have YAML frontmatter (already clean)"
    exit 0
fi

# Find where YAML ends (second ---)
YAML_END=$(grep -n "^---$" README.md | sed -n '2p' | cut -d: -f1)

if [ -z "$YAML_END" ]; then
    echo "⚠️  Warning: Could not find end of YAML frontmatter"
    exit 1
fi

# Create temporary backup for processing
TEMP_BACKUP="README.md.backup"
cp README.md "$TEMP_BACKUP"

# Remove YAML frontmatter (lines 1 to YAML_END+1 to include blank line)
sed "1,$((YAML_END+1))d" "$TEMP_BACKUP" > README.md

# Clean up temporary backup
rm -f "$TEMP_BACKUP"

echo "✅ Removed YAML frontmatter from README.md"

