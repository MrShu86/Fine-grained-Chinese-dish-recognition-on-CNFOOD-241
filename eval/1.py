import torch
from collections import Counter

ckpt_path = r"C:\Users\11138\Downloads\best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

print("Top-level type:", type(ckpt))
if isinstance(ckpt, dict):
    print("Top-level keys:", list(ckpt.keys())[:50])

# 找 state_dict
sd = None
if isinstance(ckpt, dict):
    for k in ["state_dict","model","net","module","model_state","ema_state","model_ema","backbone"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            sd = ckpt[k]; print("Use sd key:", k); break
if sd is None:
    sd = ckpt

keys = list(sd.keys())
print("Num params tensors:", len(keys))
print("First 30 keys:\n", "\n".join(keys[:30]))
print("Last 30 keys:\n", "\n".join(keys[-30:]))
import torch
ckpt = torch.load(r"C:\Users\11138\Downloads\best.pt", map_location="cpu", weights_only=False)
opt = ckpt.get("opt", None)
print(type(opt))
if isinstance(opt, dict):
    print("opt keys:", opt.keys())
    # 常见：opt['param_groups'] 里有 lr/weight_decay
    if "param_groups" in opt:
        print("lr:", opt["param_groups"][0].get("lr"))
        print("weight_decay:", opt["param_groups"][0].get("weight_decay"))
print("epoch:", ckpt.get("epoch"), "best_top1:", ckpt.get("best_top1"))
ckpt = torch.load(r"C:\Users\11138\Downloads\best.pt", map_location="cpu", weights_only=False)
pg = ckpt["opt"]["param_groups"][0]
print(pg.keys())
for k in ["betas", "eps", "momentum", "nesterov"]:
    if k in pg:
        print(k, "=", pg[k])
# 看看有哪些“顶层前缀”
prefix = [k.split(".")[0] for k in keys]
print("Top prefixes:", Counter(prefix).most_common(30))

# 专门找分类头相关
for k in keys:
    if ("head" in k) or ("classifier" in k) or k.endswith("fc.weight") or k.endswith("fc.bias"):
        v = sd[k]
        shp = tuple(v.shape) if hasattr(v, "shape") else None
        print("HEAD-LIKE:", k, shp)

# 找是否有元信息（很多训练脚本会存 args/cfg）
if isinstance(ckpt, dict):
    for meta_k in ["args","cfg","config","model_name","arch","img_size","epoch","best_acc","val_acc"]:
        if meta_k in ckpt:
            print(f"[META] {meta_k}:", ckpt[meta_k])