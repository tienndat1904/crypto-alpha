"""
Weekly Digest
=============
Stat-based summary of the last 7 days of trading. Sent on demand via
the Telegram /weekly command, or scheduled by the watchdog.
"""

from collections import Counter
from datetime import datetime, timezone, timedelta
from html import escape as _esc


def compute_weekly_digest(state: dict, mode_label: str = "·[bot]") -> str:
    """Return an HTML-formatted summary of the last 7 days for one mode's state."""
    history = state.get("trade_history", [])
    if not history:
        return f"📅 <b>{mode_label} WEEKLY</b>\nChưa có lệnh nào để phân tích."

    now = datetime.now(timezone.utc)
    week_start = now - timedelta(days=7)
    week_trades = []
    for tr in history:
        ca = tr.get("closed_at")
        if not ca:
            continue
        try:
            ts = datetime.fromisoformat(ca)
            if ts >= week_start:
                week_trades.append(tr)
        except Exception:
            continue

    if not week_trades:
        return f"📅 <b>{mode_label} WEEKLY</b>\nKhông có lệnh nào trong 7 ngày qua."

    n = len(week_trades)
    wins = [t for t in week_trades if t.get("pnl_usd", 0) > 0]
    losses = [t for t in week_trades if t.get("pnl_usd", 0) <= 0]
    total_pnl = sum(t.get("pnl_usd", 0) for t in week_trades)
    wr = len(wins) / n * 100 if n else 0
    avg_win = sum(t["pnl_usd"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl_usd"] for t in losses) / len(losses) if losses else 0
    profit_factor = (
        sum(t["pnl_usd"] for t in wins) / abs(sum(t["pnl_usd"] for t in losses))
        if losses and sum(t["pnl_usd"] for t in losses) != 0 else None
    )
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else None

    # Best / worst trades
    best = max(week_trades, key=lambda t: t.get("pnl_usd", 0))
    worst = min(week_trades, key=lambda t: t.get("pnl_usd", 0))

    # Per-symbol breakdown
    sym_pnl = Counter()
    sym_count = Counter()
    sym_wins = Counter()
    for t in week_trades:
        s = t.get("symbol", "?")
        sym_pnl[s] += t.get("pnl_usd", 0)
        sym_count[s] += 1
        if t.get("pnl_usd", 0) > 0:
            sym_wins[s] += 1

    top_sym = sorted(sym_pnl.items(), key=lambda x: x[1], reverse=True)[:3]
    bot_sym = sorted(sym_pnl.items(), key=lambda x: x[1])[:3]

    # Per-strategy breakdown
    strat_pnl = Counter()
    strat_count = Counter()
    for t in week_trades:
        s = t.get("strategy", "?")
        strat_pnl[s] += t.get("pnl_usd", 0)
        strat_count[s] += 1
    strats_sorted = sorted(strat_pnl.items(), key=lambda x: x[1], reverse=True)

    # Exit reason breakdown
    exit_reasons = Counter(t.get("reason", "?") for t in week_trades)

    # Build the message
    lines = [
        f"📅 <b>{mode_label} WEEKLY DIGEST</b>",
        f"<i>{week_start.strftime('%d/%m')} → {now.strftime('%d/%m')}</i>",
        "━━━━━━━━━━━━━━━━",
        f"<b>📊 Tổng quan</b>",
        f"  Lệnh: <code>{n}</code>  ({len(wins)}W / {len(losses)}L · WR <code>{wr:.0f}%</code>)",
        f"  PnL tuần: <code>${total_pnl:+.2f}</code>",
        f"  Avg win: <code>${avg_win:+.2f}</code>  ·  Avg loss: <code>${avg_loss:+.2f}</code>",
    ]
    if rr is not None:
        lines.append(f"  R:R: <code>{rr:.2f}</code>")
    if profit_factor is not None:
        lines.append(f"  Profit factor: <code>{profit_factor:.2f}</code>")

    lines += [
        "",
        "<b>🏆 Lệnh tốt nhất</b>",
        f"  {_esc(best.get('symbol', '?'))} {_esc(best.get('side', '').upper())} "
        f"<code>${best.get('pnl_usd', 0):+.2f}</code> ({_esc(str(best.get('reason', '?'))[:18])})",
        "<b>💀 Lệnh tệ nhất</b>",
        f"  {_esc(worst.get('symbol', '?'))} {_esc(worst.get('side', '').upper())} "
        f"<code>${worst.get('pnl_usd', 0):+.2f}</code> ({_esc(str(worst.get('reason', '?'))[:18])})",
    ]

    if top_sym:
        lines.append("")
        lines.append("<b>📈 Top 3 coin sinh lời</b>")
        for s, p in top_sym:
            wr_s = sym_wins[s] / sym_count[s] * 100
            lines.append(f"  {_esc(s)}: <code>${p:+.2f}</code>  "
                         f"({sym_count[s]} lệnh · WR {wr_s:.0f}%)")

    if bot_sym and bot_sym != top_sym:
        # Only show bottom if distinct from top (not same coin)
        bot_unique = [(s, p) for s, p in bot_sym if s not in [t[0] for t in top_sym]]
        if bot_unique:
            lines.append("")
            lines.append("<b>📉 Top 3 coin lỗ nhiều</b>")
            for s, p in bot_unique[:3]:
                wr_s = sym_wins[s] / sym_count[s] * 100
                lines.append(f"  {_esc(s)}: <code>${p:+.2f}</code>  "
                             f"({sym_count[s]} lệnh · WR {wr_s:.0f}%)")

    lines.append("")
    lines.append("<b>🎯 Theo strategy</b>")
    for s, p in strats_sorted:
        lines.append(f"  {_esc(str(s))}: <code>${p:+.2f}</code> ({strat_count[s]} lệnh)")

    lines.append("")
    lines.append("<b>🚪 Lý do thoát</b>")
    for r, c in exit_reasons.most_common():
        lines.append(f"  {_esc(str(r))}: <code>{c}</code>")

    # Insights — auto-generated observations
    insights = []
    if wr >= 60 and rr and rr < 0.5:
        insights.append("⚠️ WR cao nhưng R:R thấp → cắt lời sớm hoặc cắt lỗ chậm.")
    if wr < 40 and rr and rr > 2:
        insights.append("✅ WR thấp nhưng R:R cao → strategy trend-following, ổn.")
    if profit_factor is not None and profit_factor < 1:
        insights.append("🚨 Profit factor &lt; 1 → tuần này thua nhiều hơn thắng. Review strategy.")
    if losses and len(losses) >= 4:
        # Check loss clustering
        recent_5 = week_trades[-5:]
        recent_losses = sum(1 for t in recent_5 if t.get("pnl_usd", 0) <= 0)
        if recent_losses >= 4:
            insights.append("🚨 4/5 lệnh gần nhất thua → có thể bot đang vào thị trường xấu.")
    if top_sym and top_sym[0][1] > total_pnl * 0.7 and total_pnl > 0:
        insights.append(f"🎯 {top_sym[0][0]} đóng góp >70% PnL — phụ thuộc 1 coin, đa dạng hơn.")

    if insights:
        lines.append("")
        lines.append("<b>💡 Nhận xét tự động</b>")
        for i in insights:
            lines.append(f"  {i}")

    return "\n".join(lines)
