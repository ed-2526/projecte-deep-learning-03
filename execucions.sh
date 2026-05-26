#!/bin/bash

echo "🚀 Iniciant les execucions"

python3 main.py --architecture CRNN_Original --dataset iam --epochs 50 --batch_size 64 --patience 100 --run_name "Execució original batch 64" --learning_rate 0.001
python3 main.py --architecture CRNN_Original --dataset iam --epochs 50 --batch_size 128 --patience 100 --run_name "Execució original batch 128" --learning_rate 0.001
python3 main.py --architecture CRNN_Original --dataset iam --epochs 50 --batch_size 256 --patience 100 --run_name "Execució original batch 256" --learning_rate 0.001