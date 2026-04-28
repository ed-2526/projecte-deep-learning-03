from tqdm.auto import tqdm
import wandb
import torch

def train(model, loader, criterion, optimizer, config, device="cuda"):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    example_ct = 0  # number of examples seen
    batch_ct = 0
    
    for epoch in tqdm(range(config.epochs), desc="Epochs"):
        
        # CANVI 1: El loader d'OCR ens retorna 3 coses, no 2.
        for _, (images, targets, target_lengths) in enumerate(loader):

            loss = train_batch(images, targets, target_lengths, model, optimizer, criterion, device=device)
            example_ct += len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)


def train_batch(images, targets, target_lengths, model, optimizer, criterion, device="cuda"):
    # Passem les 3 variables a la memòria de la gràfica (GPU)
    images = images.to(device)
    targets = targets.to(device)
    target_lengths = target_lengths.to(device)
    
    # Forward pass ➡
    outputs = model(images)
    
    # CANVI 2: La CTCLoss necessita saber la mida de les prediccions de la xarxa
    batch_size = images.size(0)
    seq_len = outputs.size(0) # Longitud temporal extreta per la CNN
    
    # Creem un vector dient que totes les imatges tenen aquesta longitud seqüencial
    input_lengths = torch.full(size=(batch_size,), fill_value=seq_len, dtype=torch.long).to(device)
    
    # Calculem l'error enviant les 4 peces clau a la CTC Loss
    loss = criterion(outputs, targets, input_lengths, target_lengths)
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    # Where the magic happens (fem .item() perquè 'loss' ara és un tensor amb requeriment de gradient)
    wandb.log({"epoch": epoch, "loss": loss.item()}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss.item():.3f}")