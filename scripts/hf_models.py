"""Open-model registry (hf_models.json): list, verify, fetch.

  list            what is registered, where it lives, what uses it
  verify [name]   files present + sha256 matches the pinned value (no network)
  fetch <name>    download that model's files at the pinned commit into local_dir (Hugging Face entries only)

Only model files come down; nothing from this project is uploaded. Run with any env that has huggingface_hub
(e.g. .venv-vision\\Scripts\\python.exe scripts\\hf_models.py verify).
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = json.load(open(os.path.join(ROOT, "hf_models.json"), encoding="utf-8"))["models"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify(name):
    m = REG[name]
    if m["source"] != "huggingface":
        try:
            from importlib.metadata import version
            pkg, want = m["package"].split("==")
            got = version(pkg)
            print(f"{name}: {pkg} {got} {'ok' if got == want else f'(registry pins {want})'}")
        except Exception:
            print(f"{name}: {m['package']} not installed in this env (it lives in {m['env']})")
        return True
    ok = True
    for f in m["files"]:
        p = os.path.join(ROOT, m["local_dir"], f)
        if not os.path.exists(p):
            print(f"{name}: missing {f}"); ok = False; continue
        if f in m.get("sha256", {}):
            good = sha256(p) == m["sha256"][f]
            print(f"{name}: {f} sha256 {'ok' if good else 'MISMATCH'}"); ok &= good
    print(f"{name}: {'ok' if ok else 'NOT ok'} (pinned {m['revision'][:10]})")
    return ok


def fetch(name):
    m = REG[name]
    assert m["source"] == "huggingface", f"{name} is a {m['source']} package: pip install {m.get('package')}"
    from huggingface_hub import hf_hub_download
    for f in m["files"]:
        hf_hub_download(m["repo"], f, revision=m["revision"], local_dir=os.path.join(ROOT, m["local_dir"]))
        print("got", f)
    verify(name)


def main():
    cmd, *args = sys.argv[1:] or ["list"]
    if cmd == "list":
        for n, m in REG.items():
            where = m.get("repo") or m.get("package")
            print(f"{n:18s} {m['source']:11s} {where:34s} -> {m['app_feature']:16s} env {m['env']}")
    elif cmd == "verify":
        sys.exit(0 if all([verify(n) for n in (args or REG)]) else 1)
    elif cmd == "fetch":
        fetch(args[0])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
