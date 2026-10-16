"""Download the human fiber annotation sets used by the held-out evaluation.

paris4_eval: dl.ash2txt.org spiral_datasets/PHercParis4/eval_fibers (Sean's eval set)
PHerc0139, PHerc1451, PHerc0332: Hugging Face bucket scrollprize/datasets spiral/<scroll>/fibers
"""
import concurrent.futures as cf
import json
import re
import sys
import urllib.request
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '/workspace/fibers')
HF_TREE = 'https://huggingface.co/api/buckets/scrollprize/datasets/tree/spiral/{scroll}/fibers?recursive=false'
HF_FILE = 'https://huggingface.co/buckets/scrollprize/datasets/resolve/spiral/{scroll}/fibers/{name}'
ASH = 'https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/eval_fibers/'


def get(url, tries=5):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as error:  # retried; reported on the last attempt
            if attempt == tries - 1:
                raise RuntimeError(f'{url}: {error}')


def fetch_all(jobs):
    def one(job):
        url, dest = job
        if dest.exists() and dest.stat().st_size > 0:
            return 0
        data = get(url)
        json.loads(data)  # refuse truncated or non-JSON payloads
        dest.write_bytes(data)
        return len(data)
    with cf.ThreadPoolExecutor(16) as pool:
        return sum(pool.map(one, jobs))


def main():
    jobs = []
    for scroll in ('PHerc0139', 'PHerc1451', 'PHerc0332'):
        entries = json.loads(get(HF_TREE.format(scroll=scroll)))
        names = sorted(e['path'].rsplit('/', 1)[-1] for e in entries if e['type'] == 'file' and e['path'].endswith('.json'))
        (OUT/scroll).mkdir(parents=True, exist_ok=True)
        jobs += [(HF_FILE.format(scroll=scroll, name=n), OUT/scroll/n) for n in names]
        print(scroll, len(names), 'files', flush=True)
    listing = get(ASH).decode()
    names = sorted(set(re.findall(r'href="([^"/?]+\.json)"', listing)))
    (OUT/'paris4_eval').mkdir(parents=True, exist_ok=True)
    jobs += [(ASH + n, OUT/'paris4_eval'/n) for n in names]
    print('paris4_eval', len(names), 'files', flush=True)
    total = fetch_all(jobs)
    print(f'downloaded {total/1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    main()
