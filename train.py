from tqdm.auto import tqdm
import wandb
import torch
# CANVI NOU: Importem les funcions de traducció i càlcul d'error
from utils.utils import EarlyStopping, decode_predictions, calculate_metrics 

# La funció ja tenia el char_to_idx, perfecte!
def train(model, train_loader, val_loader, criterion, optimizer, config, char_to_idx, device="cuda"):
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
        model.eval() 
        total_val_loss = 0.0
        
        all_true_strings = []
        all_pred_strings = []
        
        # 🌟 NOU: Creem una taula buida amb les columnes que volem
        columns = ["Imatge", "Text Real (Label)", "Predicció", "Estat"]
        val_table = wandb.Table(columns=columns)
        
        with torch.no_grad():
            # 🌟 CANVI: Afegim 'enumerate' per saber per quin batch anem
            for batch_idx, (images, targets, target_lengths) in enumerate(val_loader):
                
                images = images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                
                outputs = model(images)
                
                batch_size = images.size(0)
                seq_len = outputs.size(0) 
                input_lengths = torch.full(size=(batch_size,), fill_value=seq_len, dtype=torch.long).to(device)
                
                v_loss = criterion(outputs, targets, input_lengths, target_lengths)
                total_val_loss += v_loss.item()
                
                true_strs, pred_strs = decode_predictions(outputs, targets, target_lengths, char_to_idx)
                all_true_strings.extend(true_strs)
                all_pred_strings.extend(pred_strs)
                
                # 🌟 NOU: Només agafem imatges del PRIMER batch de la validació
                # (No volem saturar WandB pujant 2.000 imatges per època)
                if batch_idx == 0:
                    # Agafem només les primeres 10 imatges del batch
                    num_samples = min(10, batch_size) 
                    for i in range(num_samples):
                        img_cpu = images[i].cpu()
                
                        # 1. Comprovem si és la CNN Original (1 canal) o la ResNet (3 canals)
                        if img_cpu.size(0) == 1:
                            # Desfem la normalització clàssica
                            img_tensor = img_cpu * 0.5 + 0.5
                        else:
                            # Desfem la normalització d'ImageNet
                            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                            img_tensor = img_cpu * std + mean
                        
                        # 2. Truc de seguretat vital: tallem qualsevol decimal rebel 
                        # perquè no surti de [0, 1] i trenqui WandB
                        img_tensor = torch.clamp(img_tensor, 0, 1)
                        
                        # 3. Creem l'objecte Imatge per a WandB
                        w_img = wandb.Image(img_tensor)
                        
                        # 4. Posem l'emoji
                        estat = "✅" if true_strs[i] == pred_strs[i] else "❌"
                        
                        # 5. Afegim la fila a la taula
                        val_table.add_data(w_img, true_strs[i], pred_strs[i], estat)
                
        # Calculem la mitjana de l'error
        avg_val_loss = total_val_loss / len(val_loader)
        val_cer, val_wer = calculate_metrics(all_true_strings, all_pred_strings)

        # ====================================================
        # 3. REGISTRE A WANDB
        # ====================================================
        wandb.log({
            "epoch": epoch,
            "Loss/Train": avg_train_loss,
            "Loss/Validation": avg_val_loss,
            "Validation/CER": val_cer,   
            "Validation/WER": val_wer,
            "Prediccions": val_table
        }, step=example_ct)
        
        # CANVI NOU: Afegim les mètriques al print
        print(f"📊 Fi de l'Època {epoch + 1}/{config.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | CER: {val_cer:.4f} | WER: {val_wer:.4f}")

        # ====================================================
        # 2. CRIDEM L'EARLY STOPPING AL FINAL DE CADA ÈPOCA
        # ====================================================
        # CANVI NOU: Ara vigila el CER (que els caràcters s'equivoquin menys) en comptes de la loss!
        early_stopping(val_cer, model)
        
        # Si el vigilant diu que hem de parar, trenquem el bucle!
        if early_stopping.early_stop:
            print("🛑 Early stopping activat! L'entrenament s'ha aturat per evitar Overfitting.")
            break
            
    # 3. UN COP ACABAT (o aturat), CARREGUEM EL MILLOR MODEL
    print("🔄 Carregant els pesos del millor model obtingut...")
    model.load_state_dict(torch.load('best_crnn_model.pth', weights_only=True))


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