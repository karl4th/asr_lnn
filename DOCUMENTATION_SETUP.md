# DREAM Documentation Setup Guide

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation page
├── installation.rst     # Installation guide
├── quickstart.rst       # Quick start tutorial
├── api.rst              # API reference
├── architecture.rst     # Architecture details
├── examples.rst         # Usage examples
├── changelog.rst        # Version history
├── requirements.txt     # Documentation dependencies
└── Makefile             # Build automation
```

## Local Development

### Install Dependencies

```bash
pip install -r docs/requirements.txt
```

### Build HTML Documentation

```bash
cd docs
make html
```

Output will be in `docs/_build/html/`

### Open Documentation

```bash
# macOS
open _build/html/index.html

# Linux
xdg-open _build/html/index.html

# Windows
start _build\html\index.html
```

## ReadTheDocs Setup

### 1. Connect Repository

1. Go to https://readthedocs.org/
2. Log in with GitHub
3. Click "Import a Project"
4. Select `dream-nn` repository

### 2. Configuration

The `.readthedocs.yaml` file is already configured:

```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.11"
sphinx:
  configuration: docs/conf.py
python:
  install:
    - method: pip
      path: .
    - requirements: docs/requirements.txt
```

### 3. Build

ReadTheDocs will automatically:
1. Install your package (`pip install .`)
2. Install documentation dependencies
3. Build Sphinx documentation
4. Host at `https://dream-nn.readthedocs.io/`

### 4. Versioning

Tag releases in git:

```bash
git tag v0.1.3
git push origin v0.1.3
```

ReadTheDocs will automatically build documentation for each tag.

## Documentation Style

### Use NumPy/Google Style Docstrings

```python
def forward(self, x, state):
    """
    Forward pass of DREAM cell.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor (batch, input_dim)
    state : DREAMState
        Previous state
        
    Returns
    -------
    h_new : torch.Tensor
        New hidden state
    state : DREAMState
        Updated state
    """
```

### ReST Syntax

```rst
Section Header
--------------

Subsection
~~~~~~~~~~

**Bold** and *italic*

.. code-block:: python

   code here

:ref:`label` — internal link
`Link <https://...>`_ — external link
```

## Updating Documentation

1. Edit `.rst` files in `docs/`
2. Rebuild: `make html`
3. Check locally
4. Commit and push
5. ReadTheDocs rebuilds automatically

## Tips

- Keep lines under 100 characters
- Use consistent formatting
- Include examples in API docs
- Update changelog for each release
- Test links with `make linkcheck`
