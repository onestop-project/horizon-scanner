# horizon-scanner
Workflows for data-driven horizon scanning for invasive alien species.

# How to install

You can install the development version of horizon-scanner from Github with:
```bash
# Clone the repository
git clone https://github.com/onestop-project/horizon-scanner.git
cd horizon-scanner

# Optional: create a virtual environment (highly recommended!)
python -m venv .venv
# Activate the environment (Windows)
.\.venv\Scripts\activate
# Activate the environment (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run all tests to verify installation
pytest --maxfail=1 --disable-warnings -q
```
