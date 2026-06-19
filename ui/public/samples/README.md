# Sample images for the ForgeVision UI

These are **synthetic placeholders** — not MVTec AD images (license prohibits redistribution).

- `normal_pattern.png` — uniform texture (use with a texture category like `carpet` or `tile`)
- `defect_pattern.png` — same base with a bright defect blob

For realistic demos, drag images from your local MVTec copy, e.g.:

```
data/mvtec_ad/bottle/test/broken_large/000.png
data/mvtec_ad/carpet/test/color/000.png
```

Generate/regenerate synthetic samples:

```powershell
cd forgevision
.\.venv\Scripts\python.exe ui/scripts/generate_samples.py
```
