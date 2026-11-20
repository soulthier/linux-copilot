import json
import os
import subprocess
from pathlib import Path

OUT_FILE = Path("data/processed/linux_man_docs.json")
MAX_DOCS = 3000


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        ["apropos", "-s", "1,5,7,8", "."],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    pages = []
    seen = set()
    for line in res.stdout.splitlines():
        if " (" not in line:
            continue
        name = line.split(" (", 1)[0].strip().split(",")[0].strip()
        sec = line.split(" (", 1)[1].split(")", 1)[0].strip()
        key = (name, sec)
        if name and key not in seen:
            seen.add(key)
            pages.append((name, sec))
        if len(pages) >= MAX_DOCS:
            break

    env = os.environ.copy()
    env.update({
        "MANWIDTH": "100",
        "MANPAGER": "cat",
        "PAGER": "cat",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    })

    docs = []
    for idx, (name, sec) in enumerate(pages, start=1):
        try:
            man_out = subprocess.run(
                ["man", sec, name],
                capture_output=True,
                text=True,
                env=env,
                timeout=5,
                check=False,
            )
        except subprocess.TimeoutExpired:
            continue

        txt = man_out.stdout.strip()
        if len(txt) < 200:
            continue

        docs.append({
            "doc_id": f"{name}_{sec}",
            "title": f"{name}({sec})",
            "source": f"linux_man_s{sec}",
            "text": txt[:8000],
        })
        if idx % 100 == 0:
            print(f"[*] Processed {idx}/{len(pages)} pages, valid docs: {len(docs)}", flush=True)

    OUT_FILE.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[+] Saved docs: {len(docs)} -> {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()