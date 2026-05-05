from tqdm.auto import tqdm
import wandb
import torch
from utils.utils import EarlyStopping

# CANVI 1: Afegim val_loader a la definició de la funció
def train(model, train_loader, val_loader, criterion, optimizer, config, device="cuda"):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # 1. INICIALITZEM L'EARLY STOPPING AQUÍ (Abans del bucle d'èpoques)
    # Paciència de 5 significa que si en 5 èpoques no millora, s'atura.
    early_stopping = EarlyStopping(patience=5, path='best_crnn_model.pth')

    # Run training and track with wandb
    example_ct = 0  # number of examples seen
    batch_ct = 0
    
    for epoch in tqdm(range(config.epochs), desc="Epochs"):
        
        # ====================================================
        # 1. FASE D'ENTRENAMENT (Train)
        # ====================================================
        model.train() # <-- SUPER IMPORTANT: Activa el Dropout i permet actualitzar pesos
        total_train_loss = 0.0
        
        for images, targets, target_lengths in train_loader:
            # Aprenem d'aquest batch
            loss = train_batch(images, targets, target_lengths, model, optimizer, criterion, device=device)
            total_train_loss += loss.item()
            example_ct += len(images)
            batch_ct += 1

            # Report metrics a la consola (opcional)
            if ((batch_ct + 1) % 25) == 0:
                print(f"Training Loss after {str(example_ct).zfill(5)} examples: {loss.item():.3f}")

        # Calculem la mitjana de l'error de Train d'aquesta època
        avg_train_loss = total_train_loss / len(train_loader)

        # ====================================================
        # 2. FASE DE VALIDACIÓ (Validation)
        # ====================================================
        model.eval() # <-- SUPER IMPORTANT: Apaga el Dropout, mode Examen.
        total_val_loss = 0.0
        
        with torch.no_grad(): # <-- SUPER IMPORTANT: No guardis gradients. Estalvia memòria i temps!
            for images, targets, target_lengths in val_loader:
                
                # Passem a la GPU
                images = images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                
                # Forward pass (predicció sense aprendre)
                outputs = model(images)
                
                # Reconstruïm les variables per la CTC Loss
                batch_size = images.size(0)
                seq_len = outputs.size(0) 
                input_lengths = torch.full(size=(batch_size,), fill_value=seq_len, dtype=torch.long).to(device)
                
                # Calculem l'error
                v_loss = criterion(outputs, targets, input_lengths, target_lengths)
                total_val_loss += v_loss.item()
                
        # Calculem la mitjana de l'error de Validació d'aquesta època
        avg_val_loss = total_val_loss / len(val_loader)

        # ====================================================
        # 3. REGISTRE A WANDB
        # ====================================================
        # Pugem TOTA la informació de l'època junta. 
        # Així veuràs com les dues línies es comparen a la web!
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        }, step=example_ct)
        
        print(f"📊 Fi de l'Època {epoch + 1}/{config.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ====================================================
        # 2. CRIDEM L'EARLY STOPPING AL FINAL DE CADA ÈPOCA
        # ====================================================
        early_stopping(avg_val_loss, model)
        
        # Si el vigilant diu que hem de parar, trenquem el bucle!
        if early_stopping.early_stop:
            print("🛑 Early stopping activat! L'entrenament s'ha aturat per evitar Overfitting.")
            break
            
    # 3. UN COP ACABAT (o aturat), CARREGUEM EL MILLOR MODEL
    print("🔄 Carregant els pesos del millor model obtingut...")
    model.load_state_dict(torch.load('best_crnn_model.pth'))


def train_batch(images, targets, target_lengths, model, optimizer, criterion, device="cuda"):
    # Passem les variables a la memòria de la gràfica (GPU)
    images = images.to(device)
    targets = targets.to(device)
    target_lengths = target_lengths.to(device)
    
    # Forward pass ➡
    outputs = model(images)
    
    # La CTCLoss necessita saber la mida de les prediccions de la xarxa
    batch_size = images.size(0)
    seq_len = outputs.size(0) # Longitud temporal extreta per la CNN
    
    # Creem un vector dient que totes les imatges tenen aquesta longitud seqüencial
    input_lengths = torch.full(size=(batch_size,), fill_value=seq_len, dtype=torch.long).to(device)
    
    # Calculem l'error enviant les 4 peces clau a la CTC Loss
    loss = criterion(outputs, targets, input_lengths, target_lengths)
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step amb l'optimizer (Aprenem!)
    optimizer.step()

    return loss