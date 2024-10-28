#!/bin/bash

if [ -d ".venv" ]; then
    echo "Activating existing virtual environment..."
    source "$PWD/.venv/bin/activate"
    echo "Using $PWD/.venv/bin/activate"
else
    echo "Creating a new virtual environment..."
    python3 -m pip install virtualenv
    python3 -m virtualenv .venv
    source .venv/bin/activate
    echo "Made $PWD/.venv/bin/activate"
    cd .revolve2-detached
    sh student_install.sh
    cd ..
fi
