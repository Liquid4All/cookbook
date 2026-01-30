import sys
import nbformat
from pathlib import Path

# Find all .ipynb files in the current directory
notebook_files = Path('./notebooks').glob('*.ipynb')

changes_made = False

for notebook_path in notebook_files:
    # Read the notebook
    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    # Remove widget metadata if present
    if 'widgets' in nb.metadata:
        del nb.metadata['widgets']

        # Save the cleaned notebook
        with open(notebook_path, 'w') as f:
            nbformat.write(nb, f)

        print(f"Removed widget metadata from: {notebook_path}")
        changes_made = True
    else:
        print(f"No widget metadata in: {notebook_path}")

# Exit with code 1 if changes were made, so pre-commit knows to re-stage
if changes_made:
    sys.exit(1)