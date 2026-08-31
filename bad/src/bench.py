import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, AutoModelForSequenceClassification
m = sys.argv[1]; L = int(sys.argv[2]); B = int(sys.argv[3])
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForSequenceClassification.from_pretrained(m, num_labels=1).cuda()
print("attn impl:", getattr(model.config, "_attn_implementation", "?"))
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
ids = torch.randint(100, 1000, (B, L)).cuda(); att = torch.ones_like(ids)
yb = torch.rand(B).cuda()
lf = torch.nn.BCEWithLogitsLoss()
for i in range(3):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = lf(model(input_ids=ids, attention_mask=att).logits.squeeze(-1).float(), yb)
    loss.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize(); t0 = time.time(); N = 15
for i in range(N):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = lf(model(input_ids=ids, attention_mask=att).logits.squeeze(-1).float(), yb)
    loss.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize()
dt = (time.time()-t0)/N
print(f"{m} L={L} B={B}: {dt*1000:.0f} ms/step  -> {278*3*dt/60:.1f} min per fold(3ep,278steps)")
