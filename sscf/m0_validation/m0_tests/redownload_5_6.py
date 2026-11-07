import urllib.request
import time
import gzip
import os

files_to_get = [5, 6]
base = 'https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v1'

for n in files_to_get:
    fname = f'vm_cpu_readings-file-{n}-of-125.csv.gz'
    url = f'{base}/trace_data_vm_cpu_readings_{fname}'
    out = os.path.join('data', fname)
    print(f"[{time.strftime('%X')}] Downloading fresh {fname}...")
    urllib.request.urlretrieve(url, out)
    print(f"[{time.strftime('%X')}] Downloaded {fname}. Verifying gzip...")
    with gzip.open(out, 'rb') as gz:
        while gz.read(1024*1024*10):
            pass
    print(f"[{time.strftime('%X')}] {fname}: VERIFIED OK!")

print("Files 5 and 6 downloaded and verified successfully!")
