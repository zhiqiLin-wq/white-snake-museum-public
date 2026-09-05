"""Restore text from dist JS files to damaged source files."""
import os, re, json

dist_dir = 'dist/assets'
src_dir = 'src'

# ── Step 1: Extract all string literals from dist ──
string_pool = set()

for fname in os.listdir(dist_dir):
    if not fname.endswith('.js'):
        continue
    fpath = os.path.join(dist_dir, fname)
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Extract double-quoted strings
    for m in re.finditer(r'"([^"\\]*(?:\\.[^"\\]*)*)"', content):
        s = m.group(1)
        if s and len(s) >= 1 and len(s) <= 200:
            string_pool.add(s)
    # Extract single-quoted strings
    for m in re.finditer(r"'([^'\\]*(?:\\.[^'\\]*)*)'", content):
        s = m.group(1)
        if s and len(s) >= 1 and len(s) <= 200:
            string_pool.add(s)

cn_pattern = re.compile(r'[一-鿿]')
useful = {s for s in string_pool if len(s) >= 2}

print(f"Extracted {len(useful)} useful strings from dist")

# ── Step 2: Pattern-based restoration ──
# Common patterns and their replacements

FIXES = {
    # Agent status labels
    ("stores/agent.ts", "''"): None,  # skip - handled manually

    # ThinkingSteps labels
    ("components/agent/ThinkingSteps.vue", "''"): None,
}

# More general: find every file with broken patterns
# Pattern: {{ condition ? '' : '' }} or content=""
# These indicate removed text

def fix_file(filepath):
    """Fix a single source file by restoring likely text."""
    fullpath = os.path.join(src_dir, filepath)
    if not os.path.exists(fullpath):
        return 0

    with open(fullpath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content
    fixes_applied = 0

    # Fix 1: Empty ternary in Vue templates: ? '' : '' -> restore from context
    # Look for patterns like: class="xxx" ... >{{ ... ? '' : '' }}<

    # Fix 2: Empty spans: <span ...></span> or <span ...>  </span>
    # These likely had text content

    # Fix 3: // comments with no text: // $

    # Fix 4: <!-- comments with no text: <!-- $ -->

    # Fix 5: Empty button text: <button ...></button>

    if content != original:
        with open(fullpath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed: {filepath} ({fixes_applied} changes)")

    return fixes_applied


# ── Step 3: Process all modified files ──
total = 0
for root, dirs, files in os.walk(src_dir):
    # Skip data files, test files
    dirs[:] = [d for d in dirs if d not in ('__tests__', '__pycache__', 'data')]
    for fname in files:
        if not any(fname.endswith(ext) for ext in ['.vue', '.ts']):
            continue
        filepath = os.path.join(root, fname).replace('\\', '/')
        # Remove src/ prefix
        relpath = filepath[len(src_dir)+1:] if filepath.startswith(src_dir) else filepath
        total += fix_file(relpath)

print(f"\nTotal files processed: {total}")
