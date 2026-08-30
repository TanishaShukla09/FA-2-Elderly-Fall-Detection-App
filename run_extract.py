"""Run extraction directly, logging to file."""
import subprocess, sys, os
os.chdir(r"C:\Users\menon\Documents\New OpenCode Project\Elderly-Fall-Detection")
LOG = "screenshots/extract_log.txt"
with open(LOG, "w") as f:
    f.write(f"EXTRACT START\n")
    f.flush()
p = subprocess.Popen(
    [sys.executable, "-u", "extract_feature.py"],
    stdout=f, stderr=subprocess.STDOUT,
)
p.wait()
with open(LOG, "a") as f:
    f.write(f"\nEXTRACT RC={p.returncode}\n")
print(f"Done. rc={p.returncode}")
