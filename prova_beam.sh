#!/bin/bash

echo "🚀 Iniciant la prova per veure si el Beam és bo o no"

python3 main.py
python3 main.py --use_beam_search --beam_width 3
python3 main.py --use_beam_search --beam_width 5
python3 main.py --use_beam_search --beam_width 10