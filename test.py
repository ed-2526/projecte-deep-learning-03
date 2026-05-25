import wandb
import torch
import jiwer

from utils.utils import decode_predictions  # Usem la mateixa funció que el train (amb Beam Search)


def test(model, test_loader, char_to_idx, device="cuda", save:bool=True, use_beam_search=True, beam_width=5):
    # Posem el model en mode avaluació (desactiva Dropout, BatchNorm, etc.)
    model.eval()
    
    correct_words = 0
    total_words = 0
    
    # Llistes per guardar TOTES les lletres del test i calcular WER/CER globals
    all_real_texts = []
    all_pred_texts = []
    
    # 🌟 NOU: Llista on guardarem les 20 imatges d'exemple per a la taula final
    dades_taula_test = []
    
    with torch.no_grad():
        # 🌟 CANVI MÍNIM: Afegim enumerate per saber si estem al primer batch
        for batch_idx, (images, targets, target_lengths) in enumerate(test_loader):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            # 1. Predicció de la xarxa
            outputs = model(images)  # Mida: [seq_len, batch_size, num_classes]
            
            # 2. Descodifiquem amb la funció de utils (Beam Search o Greedy segons config)
            # Nota: decode_predictions també reconstrueix el ground truth internament
            real_texts, pred_texts = decode_predictions(
                outputs, targets, target_lengths, char_to_idx,
                use_beam_search=use_beam_search,
                beam_width=beam_width
            )

            # 3. Recorrem els resultats del batch
            for i in range(len(pred_texts)):
                real_text = real_texts[i]
                pred_text = pred_texts[i]
                
                # Guardem per a calcular CER i WER al final
                all_real_texts.append(real_text)
                all_pred_texts.append(pred_text)
                
                # -- CÀLCUL D'ACCURACY (Paraules exactes) --
                if pred_text == real_text:
                    correct_words += 1
                total_words += 1

                # 🌟 NOU: Extraiem dades per a la taula només en el primer batch (màxim 20 imatges)
                if batch_idx == 0 and i < 20:
                    img_cpu = images[i].cpu()
                    
                    # Desfem normalització segons canals (CRNN Original 1, ResNet 3)
                    if img_cpu.size(0) == 1:
                        img_tensor = img_cpu * 0.5 + 0.5
                    else:
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_tensor = img_cpu * std + mean
                        
                    # Tallem decimals rebels perquè la foto no quedi blanca
                    img_tensor = torch.clamp(img_tensor, 0, 1)
                    
                    w_img = wandb.Image(img_tensor)
                    estat = "✅" if real_text == pred_text else "❌"
                    
                    # Afegim la fila a la nostra llista
                    dades_taula_test.append([w_img, real_text, pred_text, estat])

    # Neteja de cadenes buides per evitar que 'jiwer' peti (per si el model prediu silenci absolut)
    clean_trues = [t if len(t.strip()) > 0 else " " for t in all_real_texts]
    clean_preds = [p if len(p.strip()) > 0 else " " for p in all_pred_texts]

    # Càlcul de les mètriques finals amb jiwer
    word_accuracy = correct_words / total_words if total_words > 0 else 0
    cer = jiwer.cer(clean_trues, clean_preds)
    wer = jiwer.wer(clean_trues, clean_preds)

    decoder_mode = "Beam Search" if use_beam_search else "Greedy"
    print(f"Resultats sobre {total_words} imatges de test (Decoder: {decoder_mode}):")
    print(f" - Accuracy (Paraula exacta): {word_accuracy:.2%}")
    print(f" - CER (Character Error Rate): {cer:.4f} (Més baix és millor!)")
    print(f" - WER (Word Error Rate)     : {wer:.4f} (Més baix és millor!)")

    # 🌟 NOU: Creem l'objecte Table de WandB amb les dades que hem anat recollint
    taula_test = wandb.Table(
        columns=["Imatge", "Text Real", "Predicció", "Estat"], 
        data=dades_taula_test
    )
    
    print(f"Dades taula test {taula_test.columns}: {len(dades_taula_test)} files recollides.")

    # Pugem totes les mètriques i la taula a WandB
    wandb.log({
        "test_accuracy": word_accuracy,
        "test_CER": cer,
        "test_WER": wer,
        "Exemples_Finals_Test": taula_test  # 🌟 Pugem la taula!
    })

    if save:
        # En xarxes recurrents (RNN) amb seqüències dinàmiques, exportar a ONNX pot donar
        # molts errors d'arquitectura. El format natiu .pth de PyTorch és molt més segur per desar.
        torch.save(model.state_dict(), "model_crnn.pth")
        wandb.save("model_crnn.pth")
        print("Model desat correctament com a model_crnn.pth 💾")