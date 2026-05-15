#!/bin/bash
# Setup script para instalar dependencies localmente
# Uso: bash setup_local.sh

set -e

echo "================================"
echo "ShealtRI Local Setup"
echo "================================"
echo ""

# 1. Crear virtual environment
if [ ! -d ".venv" ]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv .venv
else
    echo "[1/5] Virtual environment already exists, skipping..."
fi

# 2. Activate venv
source .venv/bin/activate
echo "      ✓ Activated .venv"
echo ""

# 3. Upgrade pip
echo "[2/5] Upgrading pip, setuptools, wheel..."
pip install --timeout=300 --retries=10 --quiet --upgrade pip setuptools wheel

# 4. Install requirements
echo "[3/5] Installing project dependencies (this may take 10-30 min)..."
pip install --timeout=300 --retries=20 --no-cache-dir -r requirements.txt

# 5. Download spaCy model
echo ""
echo "[4/5] Downloading spaCy Spanish model..."
python -m spacy download es_core_news_md

# 6. Verify installation
echo ""
echo "[5/5] Verifying installation..."
python -c "
import sys
packages = [
    'numpy', 'scipy', 'scikit-learn', 'sentence-transformers',
    'nltk', 'spacy', 'joblib', 'chromadb', 'langchain',
    'requests', 'beautifulsoup4', 'streamlit', 'pytest', 'pydantic'
]
missing = []
for pkg in packages:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'❌ Missing packages: {missing}')
    sys.exit(1)
else:
    print('✓ All dependencies installed successfully!')
"

echo ""
echo "================================"
echo "✓ Setup complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "  1. Run smoke tests:"
echo "     python tests/smoke_test.py"
echo ""
echo "  2. Run CLI demo (no LSI, just keyword overlap):"
echo "     python cli_demo.py --query 'diabetes tipo 2'"
echo ""
echo "  3. Run actual CLI (full LSI pipeline):"
echo "     python cli.py --query 'diabetes tipo 2'"
echo ""
echo "  4. Run pytest integration tests:"
echo "     python -m pytest tests/integration/ -v"
echo ""
echo "  5. Run Streamlit UI:"
echo "     streamlit run ui/app.py"
echo ""
