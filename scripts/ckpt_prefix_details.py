import torch
p='rgb_finetune_epoch3.pth'
ckpt=torch.load(p,map_location='cpu')
if isinstance(ckpt,dict):
    for k in ('state_dict','model_state_dict','model','weights'):
        if k in ckpt and isinstance(ckpt[k],dict):
            ckpt=ckpt[k]; break
keys=list(ckpt.keys())
for pref in ['3','6']:
    print('PREFIX', pref)
    for k in keys:
        if k.split('.')[0]==pref:
            print(' ', k)
