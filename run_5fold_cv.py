# =============================================================================
# run_5fold_cv.py  —  FINAL VERSION (with UGB-SegNet 120 epoch fix)
# =============================================================================
import os, sys, glob, time, copy, random, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from ugbnet_segmentation_improved import (
    UGBSegNetImproved,
    UNetBaseline,
    AttentionUNet,
    SegNet,
    DeepLabV3Plus,
    TransUNet,
)

CFG = dict(
    dataset_root  = 'breast_ultrasound_dataset',
    ckpt_dir      = os.path.join('saved_models', 'cv'),
    results_dir   = 'results',
    figures_dir   = 'figures',
    img_size      = 224,
    batch_size    = 16,
    epochs        = 80,
    ugb_epochs    = 120,     # ★ FIX 1: UGB-SegNet gets 120 epochs
    phase1_end    = 50,
    lr            = 1e-4,
    weight_decay  = 1e-5,
    n_folds       = 5,
    seed          = 42,
    num_workers   = 0,
    threshold     = 0.5,
    patience      = 25,
)

MODEL_REGISTRY = {
    'UGB-SegNet':  lambda: UGBSegNetImproved(),
    'U-Net':       lambda: UNetBaseline(),
    'Att-UNet':    lambda: AttentionUNet(),
    'SegNet':      lambda: SegNet(),
    'DeepLabV3+':  lambda: DeepLabV3Plus(),
    'TransUNet':   lambda: TransUNet(),
}

DARK='#0d1117'; DAXC='#161b22'; DBRD='#30363d'
PAL =['#58a6ff','#3fb950','#f78166','#d2a8ff','#ffa657','#79c0ff']


def set_seed(s=CFG['seed']):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True

def dark_ax(ax):
    ax.set_facecolor(DAXC); ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_edgecolor(DBRD)
    ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

def elapsed(t0):
    s = int(time.time()-t0)
    return f'{s//3600:02d}h {(s%3600)//60:02d}m {s%60:02d}s'


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class BUSIDataset(Dataset):
    def __init__(self, img_paths, mask_paths, augment=False):
        self.imgs    = img_paths
        self.masks   = mask_paths
        self.augment = augment
        sz = CFG['img_size']
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        try:
            import albumentations as A
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=30, p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.GaussNoise(p=0.3),
                A.ElasticTransform(p=0.2),
                A.RandomResizedCrop(size=(sz, sz), scale=(0.8, 1.0), p=0.3),
            ], additional_targets={'mask': 'mask'})
            self.use_alb = True
        except ImportError:
            self.use_alb = False
            self.basic_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.3),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
            ])

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        sz  = CFG['img_size']
        img = np.array(Image.open(self.imgs[idx]).convert('RGB').resize((sz, sz)))
        msk = np.array(Image.open(self.masks[idx]).convert('L').resize((sz, sz), Image.NEAREST))
        msk = (msk > 127).astype(np.uint8)
        if self.augment:
            if self.use_alb:
                aug = self.aug(image=img, mask=msk)
                img = aug['image']; msk = aug['mask']
            else:
                pil_img = self.basic_aug(Image.fromarray(img))
                img = np.array(pil_img)
        img_t = self.to_tensor(Image.fromarray(img))
        msk_t = torch.tensor(msk, dtype=torch.float32).unsqueeze(0)
        return img_t, msk_t


def load_split_paths(dataset_root, splits):
    imgs, masks, labels = [], [], []
    for split in splits:
        for cls, lbl in [('benign', 0), ('malignant', 1)]:
            d = os.path.join(dataset_root, split, cls)
            if not os.path.isdir(d): continue
            for p in sorted(glob.glob(os.path.join(d, '*.png')) +
                            glob.glob(os.path.join(d, '*.jpg'))):
                if '_mask' in os.path.basename(p): continue
                base = os.path.splitext(p)[0]
                mp   = None
                for ext in ['_mask.png', '_mask.jpg']:
                    if os.path.exists(base + ext):
                        mp = base + ext; break
                if mp:
                    imgs.append(p); masks.append(mp); labels.append(lbl)
    return imgs, masks, np.array(labels)


def get_logit(output):
    if isinstance(output, dict):         return output['main']
    if isinstance(output, (list,tuple)): return output[0]
    return output


def compute_metrics(pred_np, mask_np, th=CFG['threshold']):
    p = (pred_np > th).astype(np.uint8).flatten()
    m = (mask_np  > th).astype(np.uint8).flatten()
    tp = int((p * m).sum()); fp = int((p*(1-m)).sum())
    fn = int(((1-p)*m).sum()); tn = int(((1-p)*(1-m)).sum())
    return dict(
        dice        = (2*tp)/(2*tp+fp+fn+1e-8),
        iou         = tp    /(tp+fp+fn+1e-8),
        sensitivity = tp    /(tp+fn+1e-8),
        specificity = tn    /(tn+fp+1e-8),
        hd95        = _hd95(pred_np, mask_np, th),
    )


def _hd95(pred, mask, th=0.5):
    pb = (pred>th).astype(np.uint8); mb = (mask>th).astype(np.uint8)
    pp = np.argwhere(pb); mp = np.argwhere(mb)
    if len(pp)==0 or len(mp)==0: return 0.0
    if len(pp)>300: pp = pp[np.random.choice(len(pp),300,replace=False)]
    if len(mp)>300: mp = mp[np.random.choice(len(mp),300,replace=False)]
    d = cdist(pp.astype(float), mp.astype(float))
    return float(np.percentile(np.concatenate([d.min(1),d.min(0)]),95))


class SegLoss(nn.Module):
    def __init__(self, boundary_w=5.0, ds_w=0.40, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.bw=boundary_w; self.dw=ds_w
        self.fa=focal_alpha; self.fg=focal_gamma

    def _dice(self, logit, mask):
        p=torch.sigmoid(logit)
        n=(2*(p*mask).sum(dim=(1,2,3)))
        d=p.sum(dim=(1,2,3))+mask.sum(dim=(1,2,3))+1e-8
        return (1-n/d).mean()

    def _focal(self, logit, mask):
        bce=nn.functional.binary_cross_entropy_with_logits(logit,mask,reduction='none')
        pt=torch.where(mask==1,torch.sigmoid(logit),1-torch.sigmoid(logit))
        w=torch.where(mask==1,torch.full_like(pt,self.fa),torch.full_like(pt,1-self.fa))
        return (w*(1-pt)**self.fg*bce).mean()

    def _iou(self, logit, mask):
        p=torch.sigmoid(logit)
        i=(p*mask).sum(dim=(1,2,3))
        u=p.sum(dim=(1,2,3))+mask.sum(dim=(1,2,3))-i+1e-8
        return (1-i/u).mean()

    def _boundary(self, logit, mask):
        k_dil = torch.ones(1, 1, 11, 11, device=logit.device)
        dil   = nn.functional.conv2d(mask, k_dil, padding=5).clamp(0, 1)
        k_ero = torch.ones(1, 1,  3,  3, device=logit.device)
        ero   = 1 - nn.functional.conv2d(1 - mask, k_ero, padding=1).clamp(0, 1)
        bnd   = (dil - ero).clamp(0, 1)
        bce   = nn.functional.binary_cross_entropy_with_logits(logit, mask, reduction='none')
        return (bnd * bce).mean()

    def forward(self, output, mask, is_ugbnet=False):
        if isinstance(output, dict):
            logit = output['main']
            if is_ugbnet:
                loss = (self._dice(logit, mask) +
                        self._focal(logit, mask) +
                        self.bw * self._boundary(logit, mask))
                for k in ['ds3', 'ds2', 'ds1']:
                    if k in output:
                        aux = output[k]
                        if aux.shape != mask.shape:
                            aux = nn.functional.interpolate(aux, mask.shape[-2:],
                                      mode='bilinear', align_corners=False)
                        loss += self.dw * (self._dice(aux, mask) + self._focal(aux, mask))
            else:
                loss = (self._dice(logit, mask) +
                        self._focal(logit, mask) +
                        self._iou(logit, mask))
        else:
            loss = self._dice(output,mask)+self._focal(output,mask)+self._iou(output,mask)
        return loss


def train_fold(model_name, model, train_ds, val_ds, device, fold, smoke=False):
    is_ugb    = (model_name == 'UGB-SegNet')
    os.makedirs(CFG['ckpt_dir'], exist_ok=True)
    save_path = os.path.join(CFG['ckpt_dir'],
                              f'{model_name.replace("+","plus")}_fold{fold}.pth')

    # ★ FIX 1: UGB-SegNet gets 120 epochs, baselines get 80
    epochs = 4 if smoke else (CFG['ugb_epochs'] if is_ugb else CFG['epochs'])

    criterion = SegLoss()
    model     = model.to(device)

    if is_ugb:
        model.freeze_backbone(n=4)
        print(f"      Phase 1: backbone FROZEN (stages 0-3, epochs 1-{CFG['phase1_end']})")

    train_ldr = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True,
                           num_workers=CFG['num_workers'],
                           pin_memory=torch.cuda.is_available())
    val_ldr   = DataLoader(val_ds,   batch_size=CFG['batch_size'], shuffle=False,
                           num_workers=CFG['num_workers'],
                           pin_memory=torch.cuda.is_available())

    opt   = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    sched = CosineAnnealingWarmRestarts(opt, T_0=epochs, T_mult=1, eta_min=1e-6)

    best_dice, best_state = 0.0, None
    no_improve = 0
    t0         = time.time()

    for ep in range(1, epochs+1):

        if is_ugb and ep == CFG['phase1_end']+1 and not smoke:
            model.unfreeze_all()
            # ★ FIX 2: Phase 2 LR = 3e-5 (was 1e-5 — too small, caused plateau)
            phase2_lr = CFG['lr'] * 0.3
            opt   = optim.AdamW(model.parameters(),
                                 lr=phase2_lr,
                                 weight_decay=CFG['weight_decay'])
            sched = CosineAnnealingWarmRestarts(
                opt, T_0=epochs-CFG['phase1_end'], T_mult=1, eta_min=1e-6)
            print(f"      Phase 2: all layers UNFROZEN "
                  f"(lr={phase2_lr:.1e}, epochs {ep}-{epochs})")

        model.train()
        tr_loss = 0.0
        for imgs, masks in train_ldr:
            imgs=imgs.to(device); masks=masks.to(device)
            opt.zero_grad()
            out  = model(imgs)
            loss = criterion(out, masks, is_ugbnet=is_ugb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        sched.step()

        if ep % 5 == 0 or ep == epochs:
            vd = _quick_dice(model, val_ldr, device)
            print(f"      Ep {ep:3d}/{epochs} | "
                  f"Loss {tr_loss/len(train_ldr):.4f} | "
                  f"Val Dice {vd*100:.2f}% | {elapsed(t0)}")
            if vd > best_dice:
                best_dice  = vd
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 5
            if no_improve >= CFG['patience'] and not smoke:
                print(f"      Early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), save_path)
    print(f"      ✓ Saved → {save_path}  (Dice: {best_dice*100:.2f}%)")
    return model, best_dice


@torch.no_grad()
def _quick_dice(model, loader, device):
    model.eval(); dices=[]
    for imgs,masks in loader:
        imgs=imgs.to(device)
        probs=torch.sigmoid(get_logit(model(imgs)))
        for b in range(imgs.size(0)):
            p=(probs[b,0].cpu().numpy()>0.5).astype(float)
            m=(masks[b,0].numpy()>0.5).astype(float)
            tp=(p*m).sum()
            dices.append(2*tp/(p.sum()+m.sum()+1e-8))
    return float(np.mean(dices))


@torch.no_grad()
def evaluate_on_test(model, test_ldr, device):
    model.eval()
    per_img = {k: [] for k in ['dice','iou','sensitivity','specificity','hd95']}
    for imgs, masks in test_ldr:
        imgs  = imgs.to(device)
        probs = torch.sigmoid(get_logit(model(imgs))).cpu().numpy()
        msk   = masks.cpu().numpy()
        for b in range(len(imgs)):
            m = compute_metrics(probs[b, 0], msk[b, 0])
            for k, v in m.items():
                per_img[k].append(v)
    means = {k: float(np.mean(v)) for k, v in per_img.items()}
    return means, per_img


def run(model_names=None, smoke=False):
    set_seed()
    for d in [CFG['ckpt_dir'],CFG['results_dir'],CFG['figures_dir']]:
        os.makedirs(d, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_names = model_names or list(MODEL_REGISTRY.keys())

    print(f"\n{'═'*68}")
    print(f"  UGB-SegNet PhD — 5-Fold Stratified CV")
    print(f"  Models : {', '.join(model_names)}")
    print(f"  Device : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type=='cuda' else ""))
    print(f"  Epochs : {4 if smoke else CFG['epochs']} (baselines) / "
          f"{4 if smoke else CFG['ugb_epochs']} (UGB-SegNet)"
          + ("  [SMOKE TEST]" if smoke else ""))
    print(f"{'═'*68}\n")

    tv_imgs,tv_masks,tv_labels = load_split_paths(CFG['dataset_root'],['train','val'])
    te_imgs,te_masks,te_labels = load_split_paths(CFG['dataset_root'],['test'])

    if len(tv_imgs)==0:
        print("  [ERROR] No images found."); return

    print(f"  Train+Val : {len(tv_imgs)} images (B:{(tv_labels==0).sum()} / M:{(tv_labels==1).sum()})")
    print(f"  Test      : {len(te_imgs)} images (B:{(te_labels==0).sum()} / M:{(te_labels==1).sum()}) <- held-out\n")

    test_ds  = BUSIDataset(te_imgs,te_masks,augment=False)
    test_ldr = DataLoader(test_ds,batch_size=CFG['batch_size'],
                          shuffle=False,num_workers=CFG['num_workers'])

    skf = StratifiedKFold(n_splits=CFG['n_folds'],shuffle=True,random_state=CFG['seed'])

    cv      = {n:{m:[] for m in ['dice','iou','sensitivity','specificity','hd95']} for n in model_names}
    per_img = {n:{m:[] for m in ['dice','iou','sensitivity','specificity','hd95']} for n in model_names}

    total_runs = len(model_names)*CFG['n_folds']
    done       = 0
    grand_t0   = time.time()

    for fold,(tr_idx,vl_idx) in enumerate(skf.split(np.zeros(len(tv_labels)),tv_labels),1):
        print(f"\n{'═'*68}")
        print(f"  FOLD {fold}/{CFG['n_folds']}  Train:{len(tr_idx)} Val:{len(vl_idx)} Test:{len(te_imgs)}")
        print(f"  Train B:{(tv_labels[tr_idx]==0).sum()} M:{(tv_labels[tr_idx]==1).sum()} | "
              f"Val B:{(tv_labels[vl_idx]==0).sum()} M:{(tv_labels[vl_idx]==1).sum()}")
        print(f"{'═'*68}")

        train_ds = BUSIDataset([tv_imgs[i] for i in tr_idx],[tv_masks[i] for i in tr_idx],augment=True)
        val_ds   = BUSIDataset([tv_imgs[i] for i in vl_idx],[tv_masks[i] for i in vl_idx],augment=False)

        for mname in model_names:
            done += 1
            ckpt = os.path.join(CFG['ckpt_dir'],f'{mname.replace("+","plus")}_fold{fold}.pth')
            print(f"\n  [{done:2d}/{total_runs}] {'★' if mname=='UGB-SegNet' else ' '} {mname}  Fold {fold}")
            print(f"  {'─'*58}")

            set_seed(CFG['seed'] + fold*31)
            model = MODEL_REGISTRY[mname]()

            if os.path.exists(ckpt):
                print(f"  [RESUME] Loading: {ckpt}")
                model.load_state_dict(torch.load(ckpt,map_location=device))
                model = model.to(device)
            else:
                fold_t0 = time.time()
                model,best_d = train_fold(mname,model,train_ds,val_ds,device,fold,smoke=smoke)
                print(f"  Done in {elapsed(fold_t0)}")

            metrics, img_scores = evaluate_on_test(model, test_ldr, device)
            for k,v in metrics.items(): cv[mname][k].append(v)
            for k,vlist in img_scores.items(): per_img[mname][k].extend(vlist)

            print(f"\n  Test — {mname} Fold {fold}:")
            print(f"    Dice:{metrics['dice']*100:.2f}%  IoU:{metrics['iou']*100:.2f}%  "
                  f"Sens:{metrics['sensitivity']*100:.2f}%  Spec:{metrics['specificity']*100:.2f}%  "
                  f"HD95:{metrics['hd95']:.2f}px")

            el=time.time()-grand_t0; eta=el/done*(total_runs-done)
            h,r=divmod(int(eta),3600); m_,s_=divmod(r,60)
            print(f"    Progress {done}/{total_runs} | ETA {h:02d}h {m_:02d}m {s_:02d}s")
            _save_results(cv,model_names,interim=True)

    print(f"\n  Total time: {elapsed(grand_t0)}")
    _print_summary(cv, per_img, model_names)
    _save_results(cv, model_names, interim=False)
    _make_plots(cv, per_img, model_names)
    print(f"\n{'═'*68}")
    print(f"  5-Fold CV Complete!")
    print(f"  CSV     -> {CFG['results_dir']}/cv_results.csv")
    print(f"  Figures -> {CFG['figures_dir']}/")
    print(f"{'═'*68}\n")


def _print_summary(cv, per_img, model_names):
    print(f"\n{'═'*78}")
    print(f"  5-FOLD CV RESULTS — BUSI | UGB-SegNet vs 5 Baselines")
    print(f"{'═'*78}")
    print(f"  {'Model':<20} {'Dice':>13} {'IoU':>12} {'Sens':>12} {'Spec':>12} {'HD95':>12}")
    print(f"  {'─'*74}")
    for nm in model_names:
        d=cv[nm]; tag='★' if nm=='UGB-SegNet' else ' '
        if not d['dice']: continue
        print(f"  {tag} {nm:<19} "
              f"{np.mean(d['dice'])*100:>5.2f}+/-{np.std(d['dice'])*100:>4.2f}%  "
              f"{np.mean(d['iou'])*100:>5.2f}+/-{np.std(d['iou'])*100:>4.2f}%  "
              f"{np.mean(d['sensitivity'])*100:>5.2f}+/-{np.std(d['sensitivity'])*100:>4.2f}%  "
              f"{np.mean(d['specificity'])*100:>5.2f}+/-{np.std(d['specificity'])*100:>4.2f}%  "
              f"{np.mean(d['hd95']):>5.2f}+/-{np.std(d['hd95']):>5.2f}")
    print(f"  {'─'*74}")
    if 'UGB-SegNet' in model_names and per_img['UGB-SegNet']['dice']:
        ugb_img = np.array(per_img['UGB-SegNet']['dice'])
        print(f"\n  Wilcoxon test on per-image Dice (n={len(ugb_img)}):")
        for nm in model_names:
            if nm=='UGB-SegNet' or not per_img[nm]['dice']: continue
            bl_img = np.array(per_img[nm]['dice'])
            n = min(len(ugb_img),len(bl_img))
            try:    W,p = wilcoxon(ugb_img[:n],bl_img[:n])
            except: W,p = 0.0,1.0
            sig='p<0.05 Sig.' if p<0.05 else 'NS'
            delta=(ugb_img.mean()-bl_img.mean())*100
            print(f"    vs {nm:<18} W={W:.1f}  p={p:.4f}  {sig}  Delta={delta:+.2f}%")


def _save_results(cv,model_names,interim=False):
    rows=[]
    for nm in model_names:
        d=cv[nm]
        if not d['dice']: continue
        row={'Model':nm}
        for m in ['dice','iou','sensitivity','specificity','hd95']:
            sc=100 if m!='hd95' else 1
            arr=np.array(d[m])
            row[f'{m}_mean']=round(arr.mean()*sc,3)
            row[f'{m}_std'] =round(arr.std() *sc,3)
            for i,v in enumerate(d[m],1):
                row[f'{m}_fold{i}']=round(v*sc,3)
        rows.append(row)
    if rows:
        tag='_interim' if interim else ''
        pd.DataFrame(rows).to_csv(
            os.path.join(CFG['results_dir'],f'cv_results{tag}.csv'),index=False)


def _make_plots(cv, per_img, model_names):
    names=[n for n in model_names if cv[n]['dice']]
    short=[n.replace('DeepLabV3+','DLV3+') for n in names]
    _plot_bars(cv,names,short)
    _plot_boxplots(cv,names,short)
    _plot_heatmap(cv,names,short)
    _plot_wilcoxon(per_img,names,short)   # ★ FIX 3: pass per_img not cv
    _plot_table(cv,names,short)
    print(f"\n  Figures saved -> {CFG['figures_dir']}/")


def _plot_bars(cv,names,short):
    show=[('dice','Dice (%)'),('iou','IoU (%)'),('sensitivity','Sensitivity (%)'),('hd95','HD95 (px)')]
    fig,axes=plt.subplots(1,4,figsize=(22,6)); fig.patch.set_facecolor(DARK)
    for ai,(metric,label) in enumerate(show):
        ax=axes[ai]; dark_ax(ax); sc=100 if metric!='hd95' else 1
        mu=[np.mean(cv[n][metric])*sc for n in names]
        sd=[np.std(cv[n][metric])*sc  for n in names]
        cs=[PAL[0] if n=='UGB-SegNet' else PAL[i%len(PAL)] for i,n in enumerate(names)]
        ax.bar(np.arange(len(names)),mu,0.6,color=cs,alpha=0.85,edgecolor='white',lw=0.6,
               yerr=sd,capsize=5,error_kw=dict(ecolor='white',lw=1.8,capthick=1.8))
        for j,(m,s) in enumerate(zip(mu,sd)):
            ax.text(j,m+s+0.4,f'{m:.2f}',ha='center',va='bottom',color='white',fontsize=8,fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short,rotation=25,ha='right',color='white',fontsize=8)
        ax.set_title(label,fontweight='bold'); ax.set_ylabel(label,fontsize=9)
    plt.suptitle('5-Fold CV — UGB-SegNet (Improved) vs 5 Baselines | BUSI',
                 color='white',fontsize=13,fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CFG['figures_dir'],'cv_bars.png'),dpi=150,bbox_inches='tight',facecolor=DARK)
    plt.close(); print(f"  [OK] cv_bars.png")


def _plot_boxplots(cv,names,short):
    fig,axes=plt.subplots(1,2,figsize=(16,6)); fig.patch.set_facecolor(DARK)
    n=len(names)
    for ai,(metric,label) in enumerate([('dice','Dice (%)'),('hd95','HD95 (px)')]):
        ax=axes[ai]; dark_ax(ax); sc=100 if metric=='dice' else 1
        data=[np.array(cv[nm][metric])*sc for nm in names]
        bp=ax.boxplot(data,patch_artist=True,notch=True,
                      medianprops=dict(color='white',lw=2.5),
                      whiskerprops=dict(color='white',lw=1.5),
                      capprops=dict(color='white',lw=1.5),
                      flierprops=dict(marker='o',markerfacecolor='white',markersize=5))
        cs=[PAL[0] if nm=='UGB-SegNet' else PAL[i%len(PAL)] for i,nm in enumerate(names)]
        for patch,c in zip(bp['boxes'],cs): patch.set_facecolor(c); patch.set_alpha(0.8)
        for j,d in enumerate(data,1):
            ax.text(j,np.mean(d)+0.3,f'{np.mean(d):.2f}',ha='center',va='bottom',
                    color='white',fontsize=9,fontweight='bold')
        ugb_i=next((i+1 for i,nm in enumerate(names) if nm=='UGB-SegNet'),None)
        if ugb_i: ax.axvspan(ugb_i-0.45,ugb_i+0.45,color='#f0c000',alpha=0.07)
        ax.set_xticks(range(1,n+1)); ax.set_xticklabels(short,color='white',fontsize=9)
        ax.set_ylabel(label,fontsize=11); ax.set_title(f'Per-fold {label}',fontweight='bold')
    plt.suptitle('Per-fold Distribution — 5-Fold CV | BUSI',color='white',fontsize=13,fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CFG['figures_dir'],'cv_boxplots.png'),dpi=150,bbox_inches='tight',facecolor=DARK)
    plt.close(); print(f"  [OK] cv_boxplots.png")


def _plot_heatmap(cv,names,short):
    matrix=np.array([cv[n]['dice'] for n in names])*100
    fig,ax=plt.subplots(figsize=(10,5)); fig.patch.set_facecolor(DARK); ax.set_facecolor(DAXC)
    sns.heatmap(matrix,annot=True,fmt='.2f',cmap='YlOrRd',
                xticklabels=[f'Fold {i+1}' for i in range(CFG['n_folds'])],
                yticklabels=short,ax=ax,linewidths=0.8,linecolor=DBRD,vmin=60,vmax=100,
                annot_kws={'size':11,'weight':'bold','color':'black'})
    ax.set_title('Per-fold Dice Heatmap (%) — 5-Fold CV | BUSI',color='white',fontsize=12,fontweight='bold')
    ax.tick_params(colors='white')
    plt.setp(ax.get_xticklabels(),color='white',fontsize=10)
    plt.setp(ax.get_yticklabels(),color='white',fontsize=10,rotation=0)
    for i,row in enumerate(matrix):
        ax.text(CFG['n_folds']+0.12,i+0.5,f'mu={row.mean():.2f}%',va='center',
                color='#f0c000' if names[i]=='UGB-SegNet' else 'white',fontsize=9,fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CFG['figures_dir'],'cv_heatmap.png'),dpi=150,bbox_inches='tight',facecolor=DARK)
    plt.close(); print(f"  [OK] cv_heatmap.png")


def _plot_wilcoxon(per_img, names, short):
    if 'UGB-SegNet' not in names: return
    ugb = np.array(per_img['UGB-SegNet']['dice'])
    if len(ugb) == 0: return
    bls,ps,ws,deltas=[],[],[],[]
    for nm in names:
        if nm=='UGB-SegNet': continue
        bl=np.array(per_img[nm]['dice']); n=min(len(ugb),len(bl))
        try:    W,p=wilcoxon(ugb[:n],bl[:n])
        except: W,p=0.0,1.0
        bls.append(nm); ps.append(p); ws.append(W)
        deltas.append((ugb.mean()-bl.mean())*100)
    p_arr=np.array(ps)
    fig,ax=plt.subplots(figsize=(12,4)); fig.patch.set_facecolor(DARK); dark_ax(ax)
    cs=['#3fb950' if p<0.05 else '#f78166' for p in p_arr]
    ax.barh(bls,-np.log10(p_arr+1e-10),color=cs,alpha=0.85,edgecolor='white',lw=0.8)
    ax.axvline(-np.log10(0.05),color='#f0c000',lw=2,ls='--',label='p=0.05')
    for i,(p,W,d) in enumerate(zip(p_arr,ws,deltas)):
        sig=f'W={W:.0f}  p={p:.4f}  Sig.' if p<0.05 else f'p={p:.4f}  NS'
        ax.text(0.3,i,f'{sig}  (Delta={d:+.2f}%)',va='center',color='white',fontsize=9)
    ax.set_xlabel('-log10(p-value)',color='white',fontsize=10)
    n_img=len(ugb)
    ax.set_title(f'Wilcoxon: UGB-SegNet vs Baselines | Per-image Dice (n={n_img})',
                 color='white',fontsize=12,fontweight='bold')
    ax.legend(facecolor=DAXC,labelcolor='white',edgecolor=DBRD)
    plt.setp(ax.get_yticklabels(),color='white',fontsize=10)
    ax.set_xlim(0,max(-np.log10(p_arr+1e-10))+5)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG['figures_dir'],'cv_wilcoxon.png'),dpi=150,bbox_inches='tight',facecolor=DARK)
    plt.close(); print(f"  [OK] cv_wilcoxon.png")


def _plot_table(cv,names,short):
    rows=[]
    for nm,sn in zip(names,short):
        d=cv[nm]; row=[sn]
        for m in ['dice','iou','sensitivity','specificity','hd95']:
            sc=100 if m!='hd95' else 1
            row.append(f"{np.mean(d[m])*sc:.2f}+/-{np.std(d[m])*sc:.2f}")
        rows.append(row)
    hdrs=['Model','Dice (%)','IoU (%)','Sens (%)','Spec (%)','HD95 (px)']
    fig,ax=plt.subplots(figsize=(18,4))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK); ax.axis('off')
    tbl=ax.table(cellText=rows,colLabels=hdrs,cellLoc='center',loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.05,2.4)
    for j in range(len(hdrs)):
        tbl[(0,j)].set_facecolor('#1f6feb')
        tbl[(0,j)].set_text_props(color='white',fontweight='bold')
    for i in range(1,len(rows)+1):
        is_u=(rows[i-1][0] in ['UGB-SegNet','UGB'])
        bg='#1a3a1a' if is_u else (DAXC if i%2 else DARK)
        for j in range(len(hdrs)):
            tbl[(i,j)].set_facecolor(bg)
            tbl[(i,j)].set_text_props(
                color='#f0c000' if is_u else 'white',
                fontweight='bold' if is_u else 'normal')
    ax.set_title('5-Fold CV — Mean+/-Std | BUSI (647 images) | UGB-SegNet vs 5 Baselines',
                 color='white',fontsize=11,fontweight='bold',pad=14)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG['figures_dir'],'04_seg_results_table.png'),
                dpi=150,bbox_inches='tight',facecolor=DARK)
    plt.close(); print(f"  [OK] 04_seg_results_table.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    names = [args.model] if args.model else list(MODEL_REGISTRY.keys())
    run(model_names=names, smoke=args.smoke)
