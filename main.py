"""GTA VI trophy rarity forecast: an ensemble on the logit scale.

The program reads every *.in case in the input directory and writes one
<id>.out per case.  Each case holds recorded reads of the game's Exophase
trophy page (players tracked plus one earn rate per trophy) and a target
instant; the output is five quantiles of every trophy's earn rate at the
target, in percent.

Method, in short (README.md carries the longer version):

  * Exophase republishes the page in steps, so the history is first
    compressed to its distinct snapshots.  Repeated identical reads carry
    no new information and would otherwise dominate any fit.
  * Five median estimates are blended on the logit scale:
      1. persistence -- the last rate, unchanged;
      2. a weighted straight-line fit of logit(rate) against
         ln(1 + hours since launch) over the distinct snapshots, with
         recent and populous snapshots weighted most, pre-break snapshots
         (a players-tracked jump that drops almost every rate) downweighted,
         and the fitted slope shrunk toward a per-class prior slope;
      3. a ratio estimate -- the same log-time fit applied separately to
         the implied earner count (rate * players / 100) and to the
         players-tracked count, divided at the target, which models the
         dilution a growing denominator causes on its own;
      4. an anchored prior drift -- the last logit plus a per-class slope
         times the log-time distance to the target, damped as the rate
         approaches the class ceiling;
      5. the recent trend -- the slope of the last day of snapshots,
         measured on its later half, projected forward.
    A saturating curve is concave in log time, so the fitted members'
    moves are damped the further ahead they are read, their slopes are
    capped by the trophy's own level, and when the recent trend is flat
    or falling it also caps how far the blend may rise above the last
    rate: dilution that has already started is not argued away by
    launch-week momentum.
  * Trophy descriptions and tiers classify each trophy (early or late
    story, finale, grind with a named count, completist, economy,
    incidental, the 100% trophy, the platinum).  The class supplies the
    prior slope, a soft ceiling the median may not run far past, and a
    spread multiplier.  The platinum is capped by the other trophies and
    the story chapters are made non-increasing in completion order,
    because earning a later chapter implies the earlier ones.
  * The spread around the median combines a horizon term growing with
    the square root of the days ahead, the disagreement between the four
    estimates, the fit's residual scatter, slope uncertainty, a
    small-earner-count term, and a jump term when the last snapshot is
    stale (a refresh the history has not seen is then likely pending).

Everything is deterministic: fixed constants, no randomness, no clock or
network use, standard library only.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

EPS_LO = 0.01
EPS_HI = 99.99


def clamp_pct(x):
    return min(EPS_HI, max(EPS_LO, x))


def logit_pct(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def pct_from_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def launch_time(case):
    try:
        day = datetime.fromisoformat(case["game"]["release_date"])
        return day.replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return parse_time(case["history"][0]["at"])


def hours_since_launch(ts, launch_ts):
    return max(0.0, (ts - launch_ts) / 3600.0)


def log_time(ts, launch_ts):
    return math.log1p(hours_since_launch(ts, launch_ts))


# ---------------------------------------------------------------------------
# Trophy classification from its own description, name and tier.
# ---------------------------------------------------------------------------

def classify(trophy):
    desc = (trophy.get("description") or "").lower()
    tier = trophy.get("tier") or "bronze"
    numbers = [int(n) for n in re.findall(r"\d[\d,]*", desc.replace(",", ""))]
    biggest = max(numbers) if numbers else 0

    info = {"ceiling": 68.0, "slope": 0.65, "spread": 1.0, "kind": "incidental"}

    if tier == "platinum" or "legend" in desc:
        info.update(ceiling=2.5, slope=0.22, spread=1.25, kind="platinum")
        return info
    if "100% completion" in desc:
        info.update(ceiling=3.5, slope=0.32, spread=1.25, kind="completion")
        return info

    chapter = re.search(r"complete chapter (\d+)", desc)
    if chapter:
        n = int(chapter.group(1))
        ceilings = {1: 93.0, 2: 87.0, 3: 80.0, 4: 72.0}
        info.update(ceiling=ceilings.get(n, 68.0),
                    slope=0.5 + 0.16 * n, spread=0.92, kind="chapter",
                    chapter=n)
        return info
    if "complete the final mission" in desc and " for " not in desc:
        info.update(ceiling=64.0, slope=1.22, spread=1.0, kind="ending")
        return info
    if "final mission for" in desc:
        info.update(ceiling=56.0, slope=0.95, spread=1.05, kind="strand")
        return info
    if desc.startswith("complete") and trophy.get("secret"):
        info.update(ceiling=76.0, slope=0.9, spread=1.0, kind="mission")
        return info
    if biggest >= 100:
        info.update(ceiling=17.0, slope=0.72, spread=1.18, kind="grind_big")
        return info
    if biggest >= 40:
        info.update(ceiling=30.0, slope=0.78, spread=1.12, kind="grind_mid")
        return info
    if biggest >= 12 or " every " in desc or desc.startswith("earn $") \
            or desc.startswith("spend $") or " all " in desc:
        info.update(ceiling=36.0, slope=0.8, spread=1.1, kind="collection")
        return info
    if tier == "gold":
        info.update(ceiling=32.0, slope=0.8, spread=1.1, kind="gold_other")
        return info
    if tier == "silver":
        info.update(ceiling=44.0, slope=0.8, spread=1.05, kind="silver_other")
        return info
    return info


# ---------------------------------------------------------------------------
# History handling: compress to distinct snapshots, mark population breaks.
# ---------------------------------------------------------------------------

def snapshots(case):
    """Return distinct snapshots: {t_end, t_start, players, rates}."""
    out = []
    for read in case["history"]:
        rates = tuple(read["rates"])
        key = (read["players_tracked"], rates)
        ts = parse_time(read["at"])
        if out and out[-1]["key"] == key:
            out[-1]["t_end"] = ts
        else:
            out.append({"key": key, "t_start": ts, "t_end": ts,
                        "players": read["players_tracked"],
                        "rates": rates})
    return out


def break_weights(snaps):
    """Downweight snapshots from before a population break.

    A break is a jump of players tracked by more than 2.5x into the next
    snapshot; the earlier population (a few hundred pre-launch testers in
    the practice data) is not the population being forecast.
    """
    weights = [1.0] * len(snaps)
    for i in range(len(snaps) - 1):
        if snaps[i]["players"] > 0 and \
                snaps[i + 1]["players"] > 2.5 * snaps[i]["players"]:
            for j in range(i + 1):
                weights[j] *= 0.25
    return weights


# ---------------------------------------------------------------------------
# Weighted straight-line fit y ~ a + b*x with slope shrinkage to a prior.
# ---------------------------------------------------------------------------

def fit_line(xs, ys, ws, prior_slope, prior_sd=0.55):
    n = len(xs)
    if n == 0:
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if n < 3 or sxx < 1e-9:
        return {"a": my - prior_slope * mx, "b": prior_slope,
                "resid": 0.35, "slope_se": prior_sd, "n": n, "mx": mx}
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b_raw = sxy / sxx
    a_raw = my - b_raw * mx
    resid2 = sum(w * (y - (a_raw + b_raw * x)) ** 2
                 for x, y, w in zip(xs, ys, ws)) / sw
    resid = math.sqrt(max(resid2, 1e-6))
    slope_se = resid / math.sqrt(sxx / sw * n) if sxx > 0 else prior_sd
    slope_se = min(max(slope_se, 0.08), 1.5)
    prec_data = 1.0 / (slope_se ** 2 + 0.15 ** 2)
    prec_prior = 1.0 / (prior_sd ** 2)
    b = (b_raw * prec_data + prior_slope * prec_prior) / (prec_data + prec_prior)
    b = min(2.3, max(-0.7, b))
    a = my - b * mx
    return {"a": a, "b": b, "resid": resid, "slope_se": slope_se,
            "n": n, "mx": mx}


# ---------------------------------------------------------------------------
# Per-case forecast.
# ---------------------------------------------------------------------------

def forecast_case(case):
    trophies = case["trophies"]
    snaps = snapshots(case)
    if not snaps:
        raise ValueError("case has no history")
    launch_ts = launch_time(case)
    checkpoint_ts = parse_time(case["at"])
    target_ts = parse_time(case["target"])
    horizon_days = max(case["horizon_minutes"], 1) / 1440.0
    x_last = log_time(snaps[-1]["t_end"], launch_ts)
    x_target = log_time(target_ts, launch_ts)
    dx = max(0.05, x_target - x_last)
    bw = break_weights(snaps)
    max_players = max(s["players"] for s in snaps) or 1

    # Denominator fit, shared by every trophy: log players vs log time,
    # recency-weighted like the trophy fits below so the earners/players
    # ratio stays coherent.
    xs_all = [log_time(s["t_end"], launch_ts) for s in snaps]
    d_fit = fit_line(xs_all,
                     [math.log(max(s["players"], 1)) for s in snaps],
                     [math.exp(-(xs_all[-1] - x) / 0.9) for x in xs_all],
                     prior_slope=0.55, prior_sd=0.4)
    d_pred = math.exp(d_fit["a"] + d_fit["b"] * x_target) if d_fit else max_players

    # How stale is the final snapshot?  Its signature first appeared at
    # t_start; a long unchanged stretch means a refresh is likely pending.
    stale_hours = max(0.0, (checkpoint_ts - snaps[-1]["t_start"]) / 3600.0)
    span_days = max(0.05, (snaps[-1]["t_end"] - snaps[0]["t_start"]) / 86400.0)

    infos = [classify(t) for t in trophies]
    medians = []
    sigmas = []

    for idx, trophy in enumerate(trophies):
        info = infos[idx]
        # Last observed rate for this trophy (search backwards).
        last_rate = None
        for s in reversed(snaps):
            if s["rates"][idx] is not None:
                last_rate = s["rates"][idx]
                break
        if last_rate is None:
            last_rate = 0.05
        last_rate = clamp_pct(last_rate)
        lg_last = logit_pct(last_rate)
        earners_last = last_rate / 100.0 * snaps[-1]["players"]

        # Points for the direct and ratio fits.  A saturating curve is
        # concave in log time -- steep on launch night, flat later -- so
        # snapshots are weighted by recency in log time as well as by
        # the population they describe: the local slope near the
        # checkpoint, not the launch-night rush, is what extrapolates.
        xs, ys, ws, ns_, ts_ = [], [], [], [], []
        for si, s in enumerate(snaps):
            r = s["rates"][idx]
            if r is None:
                continue
            x = log_time(s["t_end"], launch_ts)
            w = math.sqrt(max(s["players"], 1) / max_players) * bw[si]
            w *= math.exp(-(x_last - x) / 0.9)
            xs.append(x)
            ys.append(logit_pct(r))
            ws.append(w)
            ns_.append(max(r, 0.0) / 100.0 * s["players"])
            ts_.append(s["t_end"])

        # A straight line in log time is only a local approximation to a
        # saturating curve: the further ahead it is read, the more its
        # move from the last read is damped toward persistence, and the
        # slope itself is capped by level -- a rate already at 60% has
        # little headroom left to gain logit quickly, while one at 2%
        # can multiply as its earners arrive.
        damp = 1.0 / (1.0 + 0.45 * dx)
        slope_cap = 0.55 + 1.15 * math.exp(-last_rate / 25.0)

        def cap_delta(delta):
            return min(slope_cap * dx, max(-0.9 * dx, delta))

        direct = fit_line(xs, ys, ws, prior_slope=info["slope"])
        if direct:
            anchor = 0.5 * lg_last + 0.5 * (direct["a"] + direct["b"] * x_last)
            lg_direct = anchor + cap_delta(direct["b"] * dx) * damp
        else:
            lg_direct = lg_last

        # Ratio estimate: fit log1p(earners), divide by fitted players.
        lg_ratio = lg_last
        ratio_ok = False
        if len(xs) >= 3 and earners_last >= 8:
            n_fit = fit_line(xs, [math.log1p(v) for v in ns_], ws,
                             prior_slope=0.9, prior_sd=0.6)
            if n_fit is not None and d_fit is not None:
                n_pred = math.expm1(n_fit["a"] + n_fit["b"] * x_target)
                rate_ratio = 100.0 * max(n_pred, 0.0) / max(d_pred, 1.0)
                delta = cap_delta(logit_pct(rate_ratio) - lg_last)
                lg_ratio = lg_last + delta * damp
                ratio_ok = True

        # Anchored prior drift, damped near the class ceiling.  The
        # effective ceiling respects the class prior but never sits far
        # above a rate the trophy has actually reached, and never allows
        # more than about half the remaining headroom on top of it.
        ceiling_eff = max(info["ceiling"],
                          min(98.5, last_rate * 1.3 + 4.0))
        ceiling_eff = min(ceiling_eff,
                          max(last_rate + 6.0,
                              last_rate + (100.0 - last_rate) * 0.55))
        headroom = max(0.12, 1.0 - 0.72 * last_rate / ceiling_eff)
        lg_prior = lg_last + info["slope"] * dx * headroom

        # Recent trend: the slope over the last day of snapshots, taken
        # from its later half when there are points enough, because a
        # saturating curve decelerates -- the slope at the checkpoint,
        # not the slope averaged over the day, is what extrapolates.
        # It enters the ensemble as a fifth member and also guards it:
        # when it is flat or negative (dilution by new tracked players
        # outpacing new earners), no launch-week momentum in the older
        # fits or the prior may push the median far above its path.
        recent = [(x, y) for x, y, t in zip(xs, ys, ts_)
                  if t >= snaps[-1]["t_end"] - 86400.0]
        r_slope = None
        lg_recent = lg_last
        if len(recent) >= 2 and recent[-1][0] - recent[0][0] > 0.05:
            window = recent
            if len(recent) >= 4:
                mid_x = 0.5 * (recent[0][0] + recent[-1][0])
                later = [p for p in recent if p[0] >= mid_x]
                if len(later) >= 2 and later[-1][0] - later[0][0] > 0.03:
                    window = later
            if len(window) == 2:
                r_slope = (window[1][1] - window[0][1]) / \
                    (window[1][0] - window[0][0])
            else:
                r_fit = fit_line([p[0] for p in window],
                                 [p[1] for p in window],
                                 [1.0] * len(window),
                                 prior_slope=0.0, prior_sd=2.0)
                r_slope = r_fit["b"] if r_fit else 0.0
            if r_slope is not None:
                r_slope = min(2.0, max(-1.5, r_slope))
                lg_recent = lg_last + r_slope * dx * damp

        # Ensemble weights, adapted to how much evidence the fits carry.
        w_pers, w_dir, w_ratio, w_prior, w_rec = 0.16, 0.30, 0.20, 0.14, 0.20
        if len(snaps) < 3 or len(xs) < 3:
            w_pers, w_dir, w_ratio, w_prior, w_rec = 0.45, 0.10, 0.10, 0.35, 0.0
        if r_slope is None:
            w_dir += w_rec * 0.5
            w_pers += w_rec * 0.5
            w_rec = 0.0
        if not ratio_ok:
            w_dir += w_ratio * 0.6
            w_pers += w_ratio * 0.4
            w_ratio = 0.0
        if direct is not None and direct["resid"] > 0.6:
            move = w_dir * 0.3
            w_dir -= move
            w_pers += move
        total_w = w_pers + w_dir + w_ratio + w_prior + w_rec
        parts = [(lg_last, w_pers), (lg_direct, w_dir),
                 (lg_ratio, w_ratio), (lg_prior, w_prior),
                 (lg_recent, w_rec)]
        lg_med = sum(v * w for v, w in parts) / total_w
        disagree = math.sqrt(
            sum(w * (v - lg_med) ** 2 for v, w in parts) / total_w)

        if r_slope is not None and r_slope < 0.6:
            cap_move = max(r_slope, -1.2) * dx * damp + 0.22
            if lg_med > lg_last + cap_move:
                lg_med = lg_last + cap_move + \
                    (lg_med - lg_last - cap_move) * 0.2

        # Keep the move from the last read inside a horizon-scaled range.
        # The upward room shrinks as the rate itself grows: a trophy at
        # 50% cannot gain logit as fast as one at 2%, whose earners are
        # still arriving, while a low trophy can multiply in a week.
        up = (0.30 + 1.05 * dx) * (1.0 - 0.35 * math.sqrt(last_rate / 100.0))
        down = 0.55 + 0.45 * dx
        lg_med = min(lg_last + up, max(lg_last - down, lg_med))

        # Soft ceiling and floor: compress overshoot rather than clip.
        ceil_lg = logit_pct(ceiling_eff)
        if lg_med > ceil_lg:
            lg_med = ceil_lg + (lg_med - ceil_lg) * 0.35
        floor_lg = logit_pct(0.02)
        if lg_med < floor_lg:
            lg_med = floor_lg + (lg_med - floor_lg) * 0.5

        # Spread: horizon, disagreement, fit scatter, slope doubt,
        # small counts and a pending-refresh jump term.
        base = 0.44 * math.sqrt(horizon_days) * info["spread"]
        resid_c = min(0.5, (direct["resid"] if direct else 0.35) * 0.55)
        slope_c = min(0.5, (direct["slope_se"] if direct else 0.4) * dx)
        count_c = min(0.45, 1.15 / math.sqrt(max(earners_last, 1.0)))
        jump_c = 0.0
        if stale_hours > 2.0:
            jump_c = min(0.6, 0.2 * math.log2(stale_hours / 2.0 + 1.0))
        if any(b < 1.0 for b in bw):
            jump_c += 0.12
        sigma = math.sqrt(base * base + disagree * disagree * 0.45
                          + resid_c * resid_c + slope_c * slope_c
                          + count_c * count_c + jump_c * jump_c)
        sigma = min(1.85, max(0.12, sigma))

        medians.append(lg_med)
        sigmas.append(sigma)

    # Structural constraints: the story chain is non-increasing in
    # completion order, and the platinum cannot exceed any other trophy.
    chain = [i for i, inf in enumerate(infos) if inf["kind"] == "chapter"]
    chain.sort(key=lambda i: infos[i]["chapter"])
    ending = [i for i, inf in enumerate(infos) if inf["kind"] == "ending"]
    chain = chain + ending
    # Pool adjacent violators, with backtracking: sweep the chain into
    # blocks, and whenever a block's mean logit sits below the previous
    # block's, merge the two blocks and check again, so the finished
    # chain is genuinely non-increasing and not just locally patched.
    blocks = [[i] for i in chain]
    means = [medians[i] for i in chain]
    k = 0
    while k < len(blocks) - 1:
        if means[k] >= means[k + 1]:
            k += 1
            continue
        merged = blocks[k] + blocks[k + 1]
        mean = sum(medians[i] for i in merged) / len(merged)
        blocks[k:k + 2] = [merged]
        means[k:k + 2] = [mean]
        if k > 0:
            k -= 1
    for block, mean in zip(blocks, means):
        for i in block:
            medians[i] = mean
    plat = [i for i, inf in enumerate(infos) if inf["kind"] == "platinum"]
    if plat:
        p = plat[0]
        others_min = min(medians[i] for i in range(len(medians)) if i != p)
        if medians[p] > others_min - 0.4:
            medians[p] = others_min - 0.4

    rows = []
    for i, trophy in enumerate(trophies):
        qs = []
        prev = None
        for z in Z:
            q = round4(pct_from_logit(medians[i] + z * sigmas[i]))
            q = min(99.99, max(0.01, float(q)))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("levels") not in (None, LEVELS):
            raise ValueError(f"{path.name} asks for unexpected levels")
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name} forecast exceeds the size cap")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
