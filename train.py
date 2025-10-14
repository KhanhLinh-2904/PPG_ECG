import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import os
from torch.utils.data import DataLoader
from AE_model_2 import ResNet34_Autoencoder
from load_data import LoadData
from model import  Res34SimSiamNoise
from tqdm import tqdm
import matplotlib.pyplot as plt

'''HYPER PARAMS'''
BATCH_SIZE = 64
NUM_EPOCHS = 128
device = 'cuda'

LAMBDA = 1
subset = 0
DIM1, DIM2 = 512, 128
AE_CHECKPOINT = "checkpoint_epoch_177.pth"
PREDICTOR = True
LABEL_PERC=0.05
comment = ''
patience = 20
is_remove = False
os.makedirs('saved_models', exist_ok=True)


def cos_loss(p, z):
    p = F.normalize(p, dim=1)  # l2-normalize 
    z = F.normalize(z, dim=1)
    return - (p*z).sum(dim=1).mean()

def train_epoch(epoch_idx, model, cos_loss, ce_loss_fn, optimizer, train_loader, lambda_):
    with torch.autograd.set_detect_anomaly(True):
        train_loss = 0
        simsiam_losses = 0
        ce_losses = 0
        ECG_f1s = 0
        PPG_f1s = 0

        loop = tqdm(train_loader, desc=f"[TRAIN] Epoch {epoch_idx}", leave=False)
        for batch_idx, data in enumerate(loop):

            ECG, PPG, target = data
            ECG = ECG.unsqueeze(1).cuda()
            PPG = PPG.unsqueeze(1).cuda()
            target = target.cuda()
            target = target.float().unsqueeze(1)
            z1, z2, p1, p2, ECG_pred, PPG_pred = model(ECG, PPG)

            simsiam_loss = cos_loss(p1, z1) / 2 + cos_loss(p2, z2) / 2
            ecg_ce_loss = ce_loss_fn(ECG_pred, target)
            ppg_ce_loss = ce_loss_fn(PPG_pred, target)
        
            # total_loss = ecg_ce_loss + ppg_ce_loss 
            total_loss =   simsiam_loss

            train_loss += total_loss.item()
           

            model.zero_grad()
            total_loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # optimizer.step()
            # PPG_predicted = (PPG_pred > 0.5).int()
            # ECG_predicted = (ECG_pred > 0.5).int()

            # ECG_f1 = f1_score(target.detach().cpu().numpy(), ECG_predicted.argmax(1).detach().cpu().numpy())
            # PPG_f1 = f1_score(target.detach().cpu().numpy(), PPG_predicted.argmax(1).detach().cpu().numpy())
            # ECG_f1s += ECG_f1
            # PPG_f1s += PPG_f1

            # loop.set_postfix(loss=train_loss/(batch_idx+1), ECG_F1=ECG_f1s/(batch_idx+1), PPG_F1=PPG_f1s/(batch_idx+1))

        return train_loss / (batch_idx + 1)
    

def eval_epoch(epoch_idx, model, cos_loss, ce_loss_fn, val_loader, lambda_):

    with torch.no_grad():
        val_loss = 0
        simsiam_losses = 0
        ce_losses = 0

        PPG_preds = None
        ECG_preds = None
        all_targets = None

        model.eval()
        loop = tqdm(val_loader, desc=f"[VAL] Epoch {epoch_idx}", leave=False)
        for batch_idx, data in enumerate(loop):
            ECG, PPG, target = data
            ECG = ECG.unsqueeze(1).cuda()
            PPG = PPG.unsqueeze(1).cuda()
            target = target.cuda()
            target = target.float().unsqueeze(1)

            z1, z2, p1, p2, ECG_pred, PPG_pred = model(ECG, PPG)
            
            simsiam_loss = cos_loss(p1, z1) / 2 + cos_loss(z2, p2) / 2
            ecg_ce_loss = ce_loss_fn(ECG_pred, target)
            ppg_ce_loss = ce_loss_fn(PPG_pred, target)
           
            # total_loss = ecg_ce_loss + ppg_ce_loss 
            total_loss =  simsiam_loss

            val_loss += total_loss.item()
        #     PPG_predicted = (PPG_pred > 0.5).int()
        #     ECG_predicted = (ECG_pred > 0.5).int()
         
        #     if PPG_preds is None:
        #         PPG_preds = PPG_predicted
        #         ECG_preds = ECG_predicted
        #         all_targets = target
        #     else:
        #         PPG_preds = torch.cat((PPG_preds, PPG_predicted))
        #         ECG_preds = torch.cat((ECG_preds, ECG_predicted))
        #         all_targets = torch.cat((all_targets, target))
        
        # val_f1_ppg = f1_score(all_targets.detach().cpu().numpy(), PPG_preds.detach().cpu().numpy())
        # val_f1_ecg = f1_score(all_targets.detach().cpu().numpy(), ECG_preds.detach().cpu().numpy())
        # print(f'[VAL] Epoch {epoch_idx} Loss: {val_loss / (batch_idx + 1)} |  ECG F1: {val_f1_ecg:.4f}, PPG F1: {val_f1_ppg:.4f}')
    return val_loss / (batch_idx + 1)


def train(num_epochs, model, cos_loss, ce_loss_fn, optimizer, train_loader, val_loader, lambda_=1):
    best_val_loss = float("inf")
    # best_train_loss = float("inf")

    train_losses, val_losses = [], []
    epochs_no_improve = 0
    for epoch_idx in range(num_epochs):
        print(f'\nEpoch {epoch_idx} training...')
        train_loss = train_epoch(epoch_idx, model, cos_loss, ce_loss_fn, optimizer, train_loader, lambda_)
        val_loss = eval_epoch(epoch_idx, model, cos_loss, ce_loss_fn, val_loader, lambda_)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("✅ Saving best model...")
            print(f"Val loss improved to {best_val_loss:.4f}")
            print(f"Train loss was {train_loss:.4f}")
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"saved_models/model_{epoch_idx}.pt")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve} epochs")
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"⏹ Early stopping at epoch {epoch_idx} (no improvement in {patience} epochs)")
            break
       

    # Plot loss curves after training
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")

    plt.savefig("loss_curve.png")
    plt.show()


if __name__=='__main__':

    '''DATALOADERS'''
    print('Creating datasets')
    train_dataset = LoadData('datasets/MIMIC_train.npz')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataset = LoadData('datasets/MIMIC_val.npz')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print('Dataset finished')
    
    model = Res34SimSiamNoise(DIM1, DIM2, single_source_mode=False)


    # if os.path.exists(AE_CHECKPOINT):
    #     print(f"Loading AE checkpoint from {AE_CHECKPOINT}")
    #     ckpt = torch.load(AE_CHECKPOINT, map_location=device)
    #     pretrained_ae = ResNet34_Autoencoder().to(device)
    #     pretrained_ae.load_state_dict(ckpt['autoencoder_state_dict'])
    #     # Copy encoders (tùy tên layer trong AFNet)
    #     try:
    #         model.encoder1.load_state_dict(pretrained_ae.encoder.state_dict())
    #         model.encoder2.load_state_dict(pretrained_ae.encoder.state_dict())

    #         # model.encoder1.eval(), model.encoder2.eval()

    #     except Exception as e:
    #         print("Warning: couldn't copy encoder weights automatically:", e)
    model = nn.DataParallel(model)
    model.to(device)

    ce_loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
 


    train(NUM_EPOCHS, model, cos_loss, ce_loss_fn, optimizer, train_loader, val_loader, lambda_=LAMBDA)
