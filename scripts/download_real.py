"""Download one DFL/Sportec open match (tracking + events) and save it to data/real/raw.

The raw XML files (about 350 MB for the tracking) are fetched with a resumable download and checked against the size the
server announces, then kept in data/real/raw_xml/ and parsed locally. The earlier version streamed the file straight into
the parser; a connection that dropped part-way gave a half file and a confusing XML error.

Usage:  python scripts\\download_real.py J03WN1
"""
import os
import sys

import requests
from kloppy import sportec
from kloppy._providers.sportec import get_IDSSE_url

match_id = sys.argv[1] if len(sys.argv) > 1 else "J03WMX"
XML_DIR = "data/real/raw_xml"


def fetch(url, dest, tries=40):
    """Resumable download; the file is complete only when its size equals Content-Length."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    total = requests.head(url, allow_redirects=True, timeout=60).headers.get("Content-Length")
    if total is None:                                   # small files: the server gives no size, so fetch the whole file at once
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            open(dest, "wb").write(r.content)
        return dest
    total = int(total)
    for attempt in range(tries):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if have == total:
            return dest
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        except Exception as e:
            print(f"  connection dropped at {have / 1e6:.0f} of {total / 1e6:.0f} MB ({e.__class__.__name__}); resuming", flush=True)
    if os.path.getsize(dest) != total:
        raise SystemExit(f"could not finish {dest}")
    return dest


paths = {}
for kind in ("meta", "event", "tracking"):
    url = get_IDSSE_url(match_id, kind)
    paths[kind] = fetch(url, f"{XML_DIR}/{os.path.basename(url)}")
    print(f"{kind:9s} ok  ({os.path.getsize(paths[kind]) / 1e6:.0f} MB)", flush=True)

tracking = sportec.load_tracking(raw_data=paths["tracking"], meta_data=paths["meta"], only_alive=False)
events = sportec.load_event(event_data=paths["event"], meta_data=paths["meta"])

print("frames:", len(tracking.records), " fps:", tracking.metadata.frame_rate)
print("teams:", [t.name for t in tracking.metadata.teams])
print("coords:", tracking.metadata.coordinate_system)

tracking.to_df().to_csv(f"data/real/raw/{match_id}_tracking.csv.gz", index=False)
events.to_df().to_csv(f"data/real/raw/{match_id}_events.csv.gz", index=False)
print("saved to data/real/raw/")
