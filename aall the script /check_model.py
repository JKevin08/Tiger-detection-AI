from ultralytics import YOLO
MODEL = "/home/dant3/Downloads/ecs-project-1/tiger-training/tiger-v1/weights/best.pt"
DATA = "/home/dant3/Downloads/ecs-project-1/tiger-yolo/data.yaml"


print("\n" + "="*60)
print("        🐅 TIGER YOLO ACCURACY TEST")
print("="*60)

print("\n[+] Loading model...")
model = YOLO(MODEL)

print("[+] Running validation on TEST dataset...\n")

results = model.val(
    data=DATA,
    split="test",
    verbose=True
)

print("\n" + "="*60)
print("                 RESULTS")
print("="*60)

m = results.box

print(f"Precision       : {m.mp:.4f}  ({m.mp*100:.2f}%)")
print(f"Recall          : {m.mr:.4f}  ({m.mr*100:.2f}%)")
print(f"mAP@50          : {m.map50:.4f}  ({m.map50*100:.2f}%)")
print(f"mAP@50-95       : {m.map:.4f}  ({m.map*100:.2f}%)")

print("="*60)

print("\nModel quality:")

if m.map >= 0.80:
    print("🟢 EXCELLENT")
elif m.map >= 0.70:
    print("🟢 GOOD")
elif m.map >= 0.50:
    print("🟡 MODERATE")
elif m.map >= 0.30:
    print("🟠 WEAK")
else:
    print("🔴 POOR")

print("\n[+] Evaluation complete.")
