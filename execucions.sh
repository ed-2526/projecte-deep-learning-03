#!/bin/bash

echo "🚀 Iniciant les execucions"

# python3 main.py --architecture CRNN_Original --dataset iam --epochs 50 --batch_size 64 --patience 100 --run_name "Execució original batch 64" --learning_rate 0.001
# python3 main.py --architecture CRNN_Original --dataset iam --epochs 50 --batch_size 128 --patience 100 --run_name "Execució original batch 128" --learning_rate 0.001
# python3 main.py --architecture CRNN_Original --dataset iam --epochs 50 --batch_size 256 --patience 100 --run_name "Execució original batch 256" --learning_rate 0.001
python3 main.py --architecture CRNN_Original --dataset iam --epochs 100 --patience 5 --no_random_rotation --no_affine --run_name "2_EarlyStopping_SenseAug"
python3 main.py --architecture CRNN_Original --dataset iam --epochs 50 --patience 100 --run_name "3_DataAugment_SenseES"
python3 main.py --architecture CRNN_Original --dataset iam --epochs 100 --patience 5 --run_name "4_ES_i_DataAugment"
# Feature Extraction (Congelat)
python3 main.py --architecture ResNet18 --dataset iam --epochs 50 --patience 5 --freeze --batch_size 128 --run_name "5A_ResNet18_FeatureExtraction" --batch_size 64 --learning_rate 0.0005
# Fine-Tuning (Obert)
python3 main.py --architecture ResNet18 --dataset iam --epochs 50 --patience 5 --batch_size 128 --run_name "5B_ResNet18_FineTuning" --batch_size 128 --learning_rate 0.0005
# Feature Extraction (Congelat)
python3 main.py --architecture ResNet34 --dataset iam --epochs 50 --patience 5 --freeze --batch_size 128 --run_name "6A_ResNet34_FeatureExtraction" --learning_rate 0.0005 --batch_size 64
# Fine-Tuning (Obert)
python3 main.py --architecture ResNet34 --dataset iam --epochs 50 --patience 5 --batch_size 128 --run_name "6B_ResNet34_FineTuning" --batch_size 128 --learning_rate 0.0005
# Només Color Jitter
python3 main.py --architecture CRNN_Original --dataset iam --epochs 50 --patience 5 --activate_color_jitter --run_name "7A_Augment_ColorJitter"
# Només Gaussian Blur
python3 main.py --architecture CRNN_Original --dataset iam --epochs 50 --patience 5 --activate_gaussian_blur --run_name "7B_Augment_GaussianBlur"
python3 main.py --architecture ResNet18 --dataset iam --epochs 50 --patience 5 --batch_size 128 --use_beam_search --beam_width 5 --run_name "8_Prova_BeamSearch"