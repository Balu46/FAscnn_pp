import os
import re

def parse_metrics(file_path):
    metrics = {}
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        content = f.read()
        
    miou_match = re.search(r'Mean IoU \(overall\):\s*([\d\.]+)', content)
    if miou_match: metrics['mIoU'] = float(miou_match.group(1)) * 100
    
    gpu_fps_match = re.search(r'FPS \(throughput\):\s*([\d\.]+)', content)
    if gpu_fps_match: metrics['GPU_FPS'] = float(gpu_fps_match.group(1))

    cpu_fps_match = re.search(r'FPS:\s*([\d\.]+)', content)
    if cpu_fps_match: metrics['CPU_FPS'] = float(cpu_fps_match.group(1))

    gflops_match = re.search(r'GFLOPs\s*\(.*?\):\s*([\d\.]+)', content)
    if gflops_match: metrics['GFLOPs'] = float(gflops_match.group(1))
    
    classes = [
        "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
        "traffic sign", "vegetation", "terrain", "sky", "person", "rider",
        "car", "truck", "bus", "train", "motorcycle", "bicycle"
    ]
    
    for cls in classes:
        match = re.search(rf'Class \d+ Name {cls}\s*:\s*([\d\.]+)', content)
        if match:
            metrics[f'class_{cls}'] = float(match.group(1)) * 100

    return metrics

def find_metrics(base_dir, d):
    dir_path = os.path.join(base_dir, d)
    if not os.path.exists(dir_path): return None
    for root, dirs, files in os.walk(dir_path):
        if 'metrics.txt' in files:
            return parse_metrics(os.path.join(root, 'metrics.txt'))
    return None

dirs = [
    "FastSCNN_1024x2048", "FastSCNN_512x1024", "FastSCNN_512x512",
    "FAscnn_pp_V18_1024x2048", "FAscnn_pp_V18_512x1024", "FAscnn_pp_V18_512x512",
    "Step1_Baseline_1024x2048", "Step2_no_attn_1024x2048", "Step3_FastAttn_1024x2048",
    "Step4_CP_1024x2048", "Step5_Weights_1024x2048", "Step6_Boosted_1024x2048"
]

base_dir = "./results"

results = {}
for d in dirs:
    res = find_metrics(base_dir, d)
    if res:
        results[d] = res

# Table 1: Ewolucja do V18 (rozdzielczość 1024x2048)
print("=== TABLE 1: Ewolucja ===")
steps = [
    ("FastSCNN (Baseline)", "FastSCNN_1024x2048"),
    ("+ Inna fuzja (BiSeNetFFM)", "Step1_Baseline_1024x2048"),
    ("+ Fast Attention", "Step3_FastAttn_1024x2048"),
    ("+ Dodanie Copy-Paste", "Step4_CP_1024x2048"),
    ("+ Class Weights (Standard)", "Step5_Weights_1024x2048"),
    ("+ Manualny Boost Klas", "Step6_Boosted_1024x2048") # or FAscnn_pp_V18_1024x2048
]
for name, d in steps:
    res = results.get(d, {})
    print(f"{name} | {res.get('mIoU', 0):.2f}\\% | {res.get('GPU_FPS', 0):.2f}")

print("\n=== TABLE 2: Rozdzielczości ===")
res_steps = [
    ("1024x2048", "FAscnn_pp_V18_1024x2048"), # Wait, should it be FastSCNN or FAscnn_pp_V18? 
    # Let's print both sets just in case.
]
print("--- FAscnn++ V18 ---")
for r, d in [("1024x2048", "FAscnn_pp_V18_1024x2048"), ("512x1024", "FAscnn_pp_V18_512x1024"), ("512x512", "FAscnn_pp_V18_512x512")]:
    res = results.get(d, {})
    print(f"{r} & {res.get('mIoU', 0):.2f}\\% & {res.get('GPU_FPS', 0):.2f} & {res.get('CPU_FPS', 0):.2f} & {res.get('GFLOPs', 0):.2f} \\\\ \\hline")
    
print("--- FastSCNN ---")
for r, d in [("1024x2048", "FastSCNN_1024x2048"), ("512x1024", "FastSCNN_512x1024"), ("512x512", "FastSCNN_512x512")]:
    res = results.get(d, {})
    print(f"{r} & {res.get('mIoU', 0):.2f}\\% & {res.get('GPU_FPS', 0):.2f} & {res.get('CPU_FPS', 0):.2f} & {res.get('GFLOPs', 0):.2f} \\\\ \\hline")


print("\n=== TABLE 3: Per Class mIoU ===")
classes = [
        "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
        "traffic sign", "vegetation", "terrain", "sky", "person", "rider",
        "car", "truck", "bus", "train", "motorcycle", "bicycle"
]
d1 = results.get("FastSCNN_1024x2048", {})
d2 = results.get("FAscnn_pp_V18_1024x2048", {})
d3 = results.get("FastSCNN_512x1024", {})
d4 = results.get("FAscnn_pp_V18_512x1024", {})

for cls in classes:
    v1 = d1.get(f'class_{cls}', 0)
    v2 = d2.get(f'class_{cls}', 0)
    v3 = d3.get(f'class_{cls}', 0)
    v4 = d4.get(f'class_{cls}', 0)
    print(f"{cls.capitalize()} & {v1:.2f}\\% & {v2:.2f}\\% & {v3:.2f}\\% & {v4:.2f}\\% \\\\ \\hline")

print("--- mIoU (Średnia) ---")
print(f"mIoU & {d1.get('mIoU', 0):.2f}\\% & {d2.get('mIoU', 0):.2f}\\% & {d3.get('mIoU', 0):.2f}\\% & {d4.get('mIoU', 0):.2f}\\% \\\\ \\hline")

