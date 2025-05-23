#!/bin/bash

if [ -d ".venv" ]; then
    echo "Activating existing virtual environment..."
    source "$PWD/.venv/bin/activate"
    echo "Using $PWD/.venv/bin/activate"
else
    echo "===== Creating a new virtual environment ====="
    python -m venv .venv
    source .venv/bin/activate
    echo "===== Made $PWD/.venv/bin/activate ====="
    echo "===== INSTALLING \`.revolve2-detached\` ====="    
    cd .revolve2-detached
    sh student_install.sh
    cd ..
    pip install -r requirements.txt
    echo "All done! You can now use `source enter.sh` to enter without re-installation."
fi
