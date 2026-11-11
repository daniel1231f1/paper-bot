from __future__ import annotations

import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

from bot import Coinbase, LiveBot, run_backtest
from settings import load

load_dotenv()
cfg, run_days = load()

api = Coinbase()
bot = LiveBot(cfg, api, run_days=run_days)
app = Flask(__name__, static_folder="static")


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/state")
def state():
    return jsonify(bot.snapshot())


@app.post("/api/pause")
def pause():
    body = request.get_json(silent=True) or {}
    bot.set_paused(bool(body.get("paused", True)))
    return jsonify(ok=True, paused=bot.paused)


@app.get("/api/backtest")
def backtest_route():
    try:
        days = max(2, min(int(request.args.get("days", 14)), 30))
    except ValueError:
        return jsonify(error="days must be an integer"), 400
    try:
        rows = api.candles(cfg.product, cfg.candle_seconds, days * 86400 // cfg.candle_seconds + cfg.min_train)
    except Exception as e:
        return jsonify(error=f"Coinbase error: {e}"), 502
    if len(rows) <= cfg.min_train + 10:
        return jsonify(error="not enough data"), 502
    return jsonify(run_backtest(cfg, rows))


if __name__ == "__main__":
    bot.start()
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5050")), debug=False, threaded=True)  # localhost only
