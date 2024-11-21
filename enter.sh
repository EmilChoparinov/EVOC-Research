#!/bin/bash

if [ -d ".venv" ]; then
    echo "Activating existing virtual environment..."
    source "$PWD/.venv/bin/activate"
    echo "Using $PWD/.venv/bin/activate"
else
    echo "===== Creating a new virtual environment ====="
    echo "MAKE SURE YOU ARE ON PROJECT ROOT!!!!"
    python3 -m venv .venv
    source .venv/bin/activate
    echo "===== Made $PWD/.venv/bin/activate ====="
    echo "===== INSTALLING \`.\` ====="
    python3 -m pip install -r requirements.txt 
    echo "===== INSTALLING \`.mulitneat-detached\` ====="
    cd .multineat-detached
    pip install .
    cd ..
    echo "===== INSTALLING \`.revolve2-detached\` ====="    
    cd .revolve2-detached
    sh student_install.sh
    cd ..
    echo "All done! You can now use `source enter.sh` to enter without re-installation."
fi
