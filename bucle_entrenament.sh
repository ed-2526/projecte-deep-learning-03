#!/bin/bash

echo "🚀 Iniciant el bucle infinit d'entrenaments (CRNN -> Res18 -> Res34)"

# El bucle 'while true' fa que no s'aturi mai fins que facis Ctrl+C
while true; do

    echo "==========================================="
    echo "🧠 Llançant 10 proves de CRNN Original"
    echo "==========================================="
    wandb agent projectexn/pytorch-demo/ng4nkrvy --count 10

    echo "==========================================="
    echo "🏎️ Llançant 10 proves de ResNet18"
    echo "==========================================="
    wandb agent projectexn/pytorch-demo/vlp635s3 --count 10

    echo "==========================================="
    echo "🚜 Llançant 10 proves de ResNet34"
    echo "==========================================="
    wandb agent projectexn/pytorch-demo/qi9omx26 --count 10

    echo "🔄 Cicle completat! Fem una pausa de 5 segons i tornem a començar..."
    sleep 5

done
