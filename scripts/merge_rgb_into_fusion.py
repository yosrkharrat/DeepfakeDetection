import torch
from pathlib import Path
from collections import defaultdict

repo = Path(__file__).resolve().parents[1]
fusion_path = repo / 'results' / 'checkpoints' / 'fusion_best.pt'
backup_path = repo / 'results' / 'checkpoints' / 'fusion_best.pt.bak'
rgb_ckpt_path = repo / 'rgb_finetune_epoch3.pth'

print('paths', fusion_path.exists(), backup_path.exists(), rgb_ckpt_path.exists())

# Load fusion base
if backup_path.exists():
    base = torch.load(backup_path, map_location='cpu')
    if isinstance(base, dict) and 'state_dict' in base and isinstance(base['state_dict'], dict):
        base_sd = base['state_dict']
    elif isinstance(base, dict) and all(isinstance(v, torch.Tensor) for v in base.values()):
        base_sd = base
    else:
        # if wrapped differently
        base_sd = base.get('model_state_dict', base)
else:
    # if no backup, try loading existing fusion file
    base = torch.load(fusion_path, map_location='cpu')
    base_sd = base.get('state_dict', base)

# Load rgb checkpoint
rgb = torch.load(rgb_ckpt_path, map_location='cpu')
if isinstance(rgb, dict):
    for k in ('state_dict','model_state_dict','model','weights'):
        if k in rgb and isinstance(rgb[k], dict):
            rgb = rgb[k]
            break
if not isinstance(rgb, dict):
    raise SystemExit('rgb ckpt format unexpected')

# Normalize keys: strip module.
def strip(d):
    if all(k.startswith('module.') for k in d.keys()):
        return {k.removeprefix('module.'):v for k,v in d.items()}
    return d

base_sd = strip(base_sd)
rgb = strip(rgb)

print('base keys', len(base_sd), 'rgb keys', len(rgb))

# Build model key shapes map
# base_sd may contain other keys; we'll only replace keys that belong to rgb_model.* or that match by suffix
shape_map = defaultdict(list)
for k,v in base_sd.items():
    shape_map[tuple(v.shape)].append(k)

replaced = 0
for rk, rv in rgb.items():
    # try direct match first
    if rk in base_sd:
        base_sd[rk] = rv
        replaced += 1
        continue
    # try suffix match after numeric prefix removal
    parts = rk.split('.',1)
    suffix = parts[1] if len(parts)>1 else parts[0]
    candidates = [k for k in base_sd.keys() if k.endswith(suffix)]
    if len(candidates)==1:
        base_sd[candidates[0]] = rv
        replaced += 1
        continue
    # try shape match
    shp = tuple(rv.shape)
    cand2 = [k for k in shape_map.get(shp, []) if k not in rgb.values()]
    if len(cand2)==1:
        base_sd[cand2[0]] = rv
        replaced += 1
        continue

print('replaced count', replaced)

# Save merged checkpoint (preserve original dict wrapper if any)
orig = torch.load(backup_path if backup_path.exists() else fusion_path, map_location='cpu')
if isinstance(orig, dict) and any(isinstance(v, torch.Tensor) for v in orig.values()):
    # assume orig is raw sd
    torch.save(base_sd, fusion_path)
else:
    # try to keep wrapper
    if isinstance(orig, dict):
        # replace known fields
        for key in ('state_dict','model_state_dict','model','weights'):
            if key in orig and isinstance(orig[key], dict):
                orig[key] = base_sd
                torch.save(orig, fusion_path)
                break
        else:
            torch.save(base_sd, fusion_path)
    else:
        torch.save(base_sd, fusion_path)

print('merged saved to', fusion_path)
