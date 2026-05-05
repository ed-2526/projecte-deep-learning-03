import wandb
import torch
import editdistance

def decode_predictions(predictions, idx_to_char):
    """
    Neteja la sortida de la CTC. 
    Elimina el token 'blank' (assumint que és el 0) i els caràcters repetits consecutius.
    Exemple: [0, G, G, 0, a, 0, t, t] -> "Gat"
    """
    decoded_texts = []
    for pred in predictions:
        text = ""
        prev_char = -1
        for p in pred:
            p = p.item()
            # Si no és el token en blanc (0) i no és exactament la mateixa lletra d'abans
            if p != 0 and p != prev_char:
                text += idx_to_char.get(p, '')
            prev_char = p
        decoded_texts.append(text)
    return decoded_texts


def test(model, test_loader, char_to_idx, device="cuda", save:bool=True):
    # Posem el model en mode avaluació (desactiva Dropout, BatchNorm, etc.)
    model.eval()
    
    # Creem el diccionari invers per passar de número a lletra
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    
    # Comptadors per l'Accuracy de paraules
    correct_words = 0
    total_words = 0
    
    # Comptadors pel CER (Character Error Rate)
    total_edit_distance = 0
    total_chars = 0
    
    with torch.no_grad():
        for images, targets, target_lengths in test_loader:
            images = images.to(device)
            
            # 1. Predicció de la xarxa
            outputs = model(images) # Mida: [seq_len, batch_size, num_classes]
            
            # 2. Agafem la lletra amb més probabilitat per a cada franja
            _, preds = torch.max(outputs, 2) 
            
            # Passem a format [batch_size, seq_len] per poder iterar fàcilment
            preds = preds.transpose(1, 0)
            
            # 3. Descodifiquem les prediccions a text llegible
            pred_texts = decode_predictions(preds, idx_to_char)
            
            # 4. Reconstruïm els textos reals (targets) per poder-los comparar
            start_idx = 0
            for i in range(len(target_lengths)):
                length = target_lengths[i].item()
                # Extraiem la seqüència numèrica d'aquesta paraula en concret
                real_target = targets[start_idx : start_idx + length].tolist()
                start_idx += length
                
                # Convertim a text
                real_text = "".join([idx_to_char.get(c, '') for c in real_target])
                pred_text = pred_texts[i]
                
                # -- CÀLCUL D'ACCURACY (Paraules) --
                if pred_text == real_text:
                    correct_words += 1
                total_words += 1

                # -- CÀLCUL DEL CER (Lletres) --
                # Mesurem la distància de Levenshtein (quants canvis calen per passar de pred_text a real_text)
                distancia = editdistance.eval(pred_text, real_text)
                total_edit_distance += distancia
                total_chars += len(real_text)

    # Càlcul de les mètriques finals
    word_accuracy = correct_words / total_words if total_words > 0 else 0
    cer = total_edit_distance / total_chars if total_chars > 0 else 0

    print(f"Resultats sobre {total_words} imatges de test:")
    print(f" - Accuracy (Paraula exacta): {word_accuracy:%}")
    print(f" - CER (Error per lletres)  : {cer:.4f} (Més baix és millor!)")
    
    # Pugem totes dues mètriques a WandB
    wandb.log({
        "test_accuracy": word_accuracy,
        "test_CER": cer
    })

    if save:
        # En xarxes recurrents (RNN) amb seqüències dinàmiques, exportar a ONNX pot donar
        # molts errors d'arquitectura. El format natiu .pth de PyTorch és molt més segur per desar.
        torch.save(model.state_dict(), "model_crnn.pth")
        wandb.save("model_crnn.pth")
        print("Model desat correctament com a model_crnn.pth")