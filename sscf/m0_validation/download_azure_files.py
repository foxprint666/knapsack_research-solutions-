import os
import subprocess
import sys
import time

DEST_DIR = r"c:\Users\ASHLEY ALLEN\Downloads\m0-gate\m0\data"
BASE_URL = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v1"

def download_file(n):
    fname = f"vm_cpu_readings-file-{n}-of-125.csv.gz"
    full_dest = os.path.join(DEST_DIR, fname)
    url = f"{BASE_URL}/trace_data_vm_cpu_readings_{fname}"
    
    # Check if already complete (~640+ MB)
    if os.path.exists(full_dest) and os.path.getsize(full_dest) > 640 * 1024 * 1024:
        print(f"[{time.strftime('%X')}] File {fname} already complete ({os.path.getsize(full_dest)/(1024*1024):.1f} MB). Skipping.")
        return True

    print(f"\n[{time.strftime('%X')}] === Downloading file {n}/13: {fname} ===")
    
    # Run curl with speed limit timeout (if speed < 10KB/s for 30s, fail & retry/resume)
    cmd = [
        "curl.exe", "-L", "-C", "-",
        "--speed-time", "30", "--speed-limit", "10240",
        "--connect-timeout", "15",
        "--retry", "5", "--retry-delay", "3",
        "-o", full_dest,
        url
    ]
    ret = subprocess.run(cmd)
    return ret.returncode == 0

def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    for n in range(7, 14):
        fname = f"vm_cpu_readings-file-{n}-of-125.csv.gz"
        full_dest = os.path.join(DEST_DIR, fname)
        
        while True:
            if os.path.exists(full_dest) and os.path.getsize(full_dest) > 640 * 1024 * 1024:
                print(f"[{time.strftime('%X')}] Verified {fname} complete ({os.path.getsize(full_dest)/(1024*1024):.1f} MB).")
                break
            sz_str = f"{os.path.getsize(full_dest)/(1024*1024):.1f} MB" if os.path.exists(full_dest) else "0 MB"
            print(f"[{time.strftime('%X')}] Resuming/downloading {fname} (current size: {sz_str})...")
            download_file(n)
            time.sleep(2)

    print(f"\n[{time.strftime('%X')}] All files 1 through 13 verified and complete!")

if __name__ == "__main__":
    main()
