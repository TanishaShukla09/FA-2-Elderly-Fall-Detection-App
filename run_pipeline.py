"""Run the full ML pipeline: extract -> train -> evaluate -> test."""
import subprocess, sys, time, os, threading
os.chdir(r"C:\Users\menon\Documents\New OpenCode Project\Elderly-Fall-Detection")
LOG = "screenshots/pipeline_log.txt"

def stream_output(proc, log, label):
    """Read stdout/stderr line by line and write to log in real-time."""
    import io
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace"):
        if "[INFO]" in line or "detected" in line or "Complete" in line or "DONE" in line:
            log.write(f"[{label}] {line}")
            log.flush()
    for line in io.TextIOWrapper(proc.stderr, encoding="utf-8", errors="replace"):
        if "Traceback" in line or "Error" in line or "FAILED" in line:
            log.write(f"[{label}-ERR] {line}")
            log.flush()

with open(LOG, "w") as log:
    log.write(f"PIPELINE START {time.strftime('%H:%M:%S')}\n")
    log.flush()
    for step, cmd in [
        ("EXTRACT", [sys.executable, "-u", "extract_feature.py"]),
        ("TRAIN", [sys.executable, "-u", "train_model.py"]),
        ("EVALUATE", [sys.executable, "-u", "evaluate_model.py"]),
        ("TEST", [sys.executable, "-u", "test_app.py"]),
    ]:
        t0 = time.time()
        log.write(f"\n{'='*60}\n{step} START {time.strftime('%H:%M:%S')}\n{'='*60}\n")
        log.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             bufsize=1, text=True, encoding="utf-8", errors="replace")
        stream_output(p, log, step)
        rc = p.wait()
        elapsed = time.time() - t0
        log.write(f"{step} DONE in {elapsed:.0f}s (rc={rc})\n")
        log.flush()
        if rc != 0:
            log.write(f"FAILED at {step} (rc={rc})\n")
            break
    log.write(f"\nPIPELINE COMPLETE {time.strftime('%H:%M:%S')}\n")
    log.flush()
