"""Record 1 s snapshots of the public Coinbase BTC-USD order book + trade flow (market data only, no keys)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import websocket  # websocket-client

OUT = Path(os.getenv("BOOK_FILE", "book.csv"))
RUN_SECONDS = float(os.getenv("RUN_SECONDS", "19800"))      # 330 min
PUBLISH_EVERY = float(os.getenv("PUBLISH_EVERY", "600"))
URL = "wss://ws-feed.exchange.coinbase.com"


def depth(bids: dict, asks: dict, mid: float, bps: float) -> float:
    lo, hi = mid * (1 - bps / 1e4), mid * (1 + bps / 1e4)
    b = sum(s for p, s in bids.items() if p >= lo)
    a = sum(s for p, s in asks.items() if p <= hi)
    return (b - a) / (b + a + 1e-12)


_last_pub = [time.time()]  # module-level: a feed reconnect must not reset the publish timer


def publish() -> None:
    subprocess.run(["bash", "scripts/publish_book.sh"], check=False)  # a failed push must not kill the recorder


def session(deadline: float, f) -> None:
    try:  # python.org builds on macOS ship without root certs; certifi fixes that, Linux runners don't need it
        import certifi
        sslopt = {"ca_certs": certifi.where()}
    except ImportError:
        sslopt = {}
    ws = websocket.create_connection(URL, timeout=20, sslopt=sslopt)
    ws.send(json.dumps({"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["level2_batch", "ticker"]}))
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    buy = sell = 0.0
    ntr = 0
    last = time.time()
    while time.time() < deadline:
        m = json.loads(ws.recv())
        t = m.get("type")
        if t == "snapshot":
            bids = {float(p): float(s) for p, s in m["bids"]}
            asks = {float(p): float(s) for p, s in m["asks"]}
        elif t == "l2update":
            for side, p, s in m["changes"]:
                d, p, s = (bids if side == "buy" else asks), float(p), float(s)
                if s == 0:
                    d.pop(p, None)
                else:
                    d[p] = s
        elif t == "ticker" and "last_size" in m:
            ntr += 1
            if m.get("side") == "buy":
                buy += float(m["last_size"])
            else:
                sell += float(m["last_size"])
        now = time.time()
        if now - last >= 1 and bids and asks:
            bb, ba = max(bids), min(asks)
            mid = (bb + ba) / 2
            f.write(f"{now:.2f},{bb},{ba},{bids[bb]:.6f},{asks[ba]:.6f},{depth(bids, asks, mid, 1):.4f},"
                    f"{depth(bids, asks, mid, 3):.4f},{depth(bids, asks, mid, 10):.4f},{buy:.8f},{sell:.8f},{ntr}\n")
            buy = sell = 0.0
            ntr = 0
            last = now
        if now - _last_pub[0] >= PUBLISH_EVERY:
            f.flush()
            _last_pub[0] = now
            publish()


def main() -> int:
    deadline = time.time() + RUN_SECONDS
    with OUT.open("a", buffering=1) as f:
        while time.time() < deadline:
            try:
                session(deadline, f)
            except Exception as e:  # reconnect on any feed hiccup; gaps are tolerated by the analysis
                print("feed error, reconnecting:", repr(e), flush=True)
                time.sleep(3)
    publish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
