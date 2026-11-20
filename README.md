# GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Josesoll/gta6-rarity-forecast/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Josesoll/gta6-rarity-forecast/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

An entry for make.fun's [gta6-trophy-rarity-forecast-01](https://make.fun/c/gta6-trophy-rarity-forecast-01). The program, `main.py` (Python 3, standard library only, deterministic, no network), reads a case file of recorded reads of Grand Theft Auto VI's Exophase trophy page — every read's players-tracked count and one earn rate per PlayStation 5 trophy — and writes, for each of the 55 trophies, five quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate at the case's target instant. In the final that target is 2026-11-26T00:00:00Z, one week after launch and about 4.5 days after the last read it is given.

## How the model works

Everything happens on the logit scale, the scale the scoring uses, so a move from 1% to 2% counts for as much as one from 33% to 50%.

**Steps, not a stream.** Exophase republishes a game's counts on its own schedule, so the recorded reads repeat the same page many times. The history is first compressed to its distinct snapshots; repeated identical reads carry no information and would otherwise dominate any fit. A snapshot from before a population break — a jump in players tracked that drops almost every rate at once, as happened at the first refresh after launch — is downweighted, because that earlier, smaller population is not the one being forecast.

**Five estimates of the median, blended.** For each trophy the program combines: persistence (the last rate, unchanged); a weighted fit of logit(rate) against ln(1 + hours since launch) over the distinct snapshots, weighted by recency and by the population behind each snapshot, with its slope shrunk toward a prior; a ratio estimate that fits the implied earner count and the players-tracked count separately and divides them at the target, which captures dilution by new tracked players on its own; an anchored prior drift; and the recent trend, the slope of the last day of snapshots measured on its later half. Because trophy curves saturate — they are concave in log time — the fitted moves are damped the further ahead they are read, slopes are capped by the trophy's own level (a rate at 60% has little headroom left; one at 2% can still multiply), and a flat or falling recent trend caps how far the blend may rise above the last rate.

**What each trophy asks of the player.** Descriptions and tiers classify every trophy: story chapters and the ending, named story missions, side-strand finales, grinds with a named count (250 stunts, 100 locations, …), completist and economy trophies, one-off incidental actions, the 100%-completion gold and the platinum. The class sets the prior slope, a soft ceiling the median may not run far past, and a spread multiplier. Two structural facts are enforced afterwards: the story chain's medians are made non-increasing in completion order (finishing Chapter 4 implies Chapter 1), using pool-adjacent-violators pooling, and the platinum's median is kept below the other trophies' (nobody holds the platinum without all of them).

**An honest spread.** The width around the median combines a horizon term growing with the square root of the days ahead, the disagreement between the five estimates, the fit's residual scatter, slope uncertainty, a small-earner-count term (a rate built on a handful of earners is fragile), and a jump term when the last snapshot is stale, meaning a refresh the history has not seen is likely pending. Quantiles are rounded to 4 decimals, clamped to 0.01–99.99 and kept non-decreasing, and the program's output is byte-for-byte identical across runs.

## Running it

```
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` file in the input directory in name order and writes one `<id>.out` per case to the output directory. With no environment variables set it reads `/input` and writes `/output`, as the grader runs it.
