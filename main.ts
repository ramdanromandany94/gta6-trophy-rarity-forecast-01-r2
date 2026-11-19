// gta6-trophy-rarity-forecast-01: calibrated trend forecaster (Node 22, standard library only).
//
// Model, per trophy, on the logit scale (rates are percents, clamped to 0.01..99.99):
//   - Time is x = ln(1 + hours since the game's release). Trophy earn rates on Exophase move in
//     steps (the page only changes when Exophase refreshes it), so the fit uses only the distinct
//     snapshots: reads where the players-tracked count or any rate changed.
//   - A least-squares line of logit(rate) against x is fitted per trophy over the recent
//     window of distinct snapshots (last 36 h), its slope shrunk toward the median slope of
//     its tier group (platinum/gold/silver/bronze), and the forecast is anchored on the last
//     distinct read: y_med = y_last + damp * b_shrunk * (x_target - x_last), with the move
//     damped and capped. Anchoring keeps a flat history forecasting persistence; the recent
//     window and the damping keep a saturating rate from being extrapolated as far as its
//     early steep rise suggests.
//   - The spread is an honest predictive std on the logit scale, combined in quadrature from:
//     the per-trophy residual std around its line (shrunk toward the pooled residual), the slope
//     uncertainty over the extrapolation distance, the refresh-jump volatility seen in the history
//     (pooled logit jumps between distinct reads, times the expected number of refreshes inside the
//     horizon), and a floor. A wider spread costs a little when the median is right and saves a lot
//     when a refresh jumps the rates, which is exactly what the practice cases punish.
//   - Quantiles: q_tau = 100 / (1 + exp(-(y_med + z_tau * s))), z = (-1.645, -0.674, 0, 0.674,
//     1.645), rounded to 4 decimals. Output rows are non-decreasing by construction.
//
// What it must do (same contract as the kit starter):
//   - read every <id>.in in the input directory (the grader runs one case at a time, with only that
//     case's file there): JSON {"schema":"makefun/gta6-trophies-case/v1","id","at","horizon_minutes",
//     "target","levels","game","trophies","history","notes"}, `trophies` lists the case's trophies
//     {id, name, description, tier, points, secret} and `history` every read before `at`, oldest
//     first, each {slot, at, players_tracked, trophies_earned, rates}, `rates` in the order of
//     `trophies` (null where that read lacked the trophy);
//   - write <id>.out to the output directory: {"trophies":[{"id":1,"q":[q05,q25,q50,q75,q95]}, ...]},
//     one row for each of the case's trophies, each id once, no other key, each q five finite numbers
//     from 0 to 100 (percent), non-decreasing; at most 65,536 bytes.
// It runs offline, with no network, under the case's time cap: everything it knows is in the case
// file.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and
// /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Pseudo-count for shrinking a trophy's slope toward its tier-group median slope. */
const SLOPE_PRIOR_N = 3;
/** Prior std of a trophy's slope, in logit units per unit of ln(1 + hours). */
const SLOPE_PRIOR_SD = 0.35;
/** Hard clamp on the (shrunk) slope, in logit units per unit of ln(1 + hours). */
const SLOPE_MIN = -1.0;
const SLOPE_MAX = 1.5;
/** Only distinct snapshots from the last RECENT_WINDOW_H hours before the last read feed the
 * slope fit: logit(rate) against ln(1 + hours) is concave (rates saturate), so a line over the
 * whole history extrapolates the early steep rise far past the target. */
const RECENT_WINDOW_H = 36;
/** The extrapolated move is damped: the recent local slope still overstates how far a
 * saturating rate travels over a multi-day horizon. */
const MOVE_DAMP = 0.6;
/** Cap on the absolute extrapolated move, in logit units. */
const MOVE_CAP = 1.2;
/** Floor on the per-trophy residual std, in logit units. */
const RESID_FLOOR = 0.12;
/** Prior refresh-jump std per refresh, in logit units, when the history shows no jump. */
const JUMP_PRIOR_SD = 0.12;
/** Cap on the pooled refresh-jump std, in logit units. */
const JUMP_CAP = 0.5;
/** Cap on the expected number of refreshes inside the horizon that the jump term counts:
 * refresh jumps are mostly the trend the fit already absorbs, not independent shocks. */
const REFRESH_COUNT_CAP = 8;
/** Prior gap between Exophase refreshes, in hours, when the history shows no distinct read. */
const REFRESH_INTERVAL_PRIOR_H = 6;
/** Floor on the total predictive spread, in logit units. */
const SPREAD_FLOOR = 0.3;
/** Extra spread floor, in logit units, independent of the horizon. */
const SPREAD_BASE = 0.2;

interface Trophy {
  id: number;
  tier: string;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface Case {
  levels: number[];
  horizon_minutes: number;
  target: string;
  game: { release_date: string };
  trophies: Trophy[];
  history: Read[];
}

const clamp = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => Math.log(clamp(pct) / (100 - clamp(pct)));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const median = (xs: number[]): number => {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** Hours from `from` to `to`, from ISO 8601 UTC strings. */
const hoursBetween = (from: string, to: string): number =>
  (Date.parse(to) - Date.parse(from)) / 3_600_000;

/**
 * Least-squares line y = a + b*x through (xs, ys). Returns the slope, the intercept and the
 * residual standard deviation (0 when fewer than 3 points or no spread in x).
 */
function fitLine(xs: number[], ys: number[]): { slope: number; intercept: number; residSd: number } {
  const n = xs.length;
  if (n === 0) return { slope: 0, intercept: 0, residSd: 0 };
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx <= 1e-12) return { slope: 0, intercept: my, residSd: 0 };
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  let rss = 0;
  for (let i = 0; i < n; i++) rss += (ys[i] - (intercept + slope * xs[i])) ** 2;
  const residSd = n > 2 ? Math.sqrt(rss / (n - 2)) : 0;
  return { slope, intercept, residSd };
}

function forecastCase(c: Case, file: string): { trophies: Array<{ id: number; q: number[] }> } {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const nT = c.trophies.length;

  // Release instant: the game's release date at 00:00 UTC; fall back to the first read.
  const release = c.game?.release_date ? `${c.game.release_date}T00:00:00Z` : c.history[0].at;
  const xTarget = Math.log(1 + Math.max(0, hoursBetween(release, c.target)));

  // Distinct snapshots: a read counts when the players-tracked count or any rate moved since the
  // previous kept read. Consecutive identical reads are one stale page, not new information.
  const snaps: Read[] = [];
  for (const r of c.history) {
    const prev = snaps[snaps.length - 1];
    if (
      prev === undefined ||
      r.players_tracked !== prev.players_tracked ||
      r.rates.some((v, i) => v !== prev.rates[i])
    ) {
      snaps.push(r);
    }
  }
  const snapX = snaps.map((r) => Math.log(1 + Math.max(0, hoursBetween(release, r.at))));

  // The slope fit uses only the recent window of distinct snapshots (see RECENT_WINDOW_H):
  // keep at least 3 snapshots even when the window is short.
  const lastSnapAt = Date.parse(snaps[snaps.length - 1].at);
  let winStart = 0;
  while (
    winStart < snaps.length - 3 &&
    lastSnapAt - Date.parse(snaps[winStart].at) > RECENT_WINDOW_H * 3_600_000
  ) {
    winStart++;
  }
  const winSnaps = snaps.slice(winStart);
  const winX = snapX.slice(winStart);

  // Per-trophy series over the recent window of distinct snapshots (null rates skipped),
  // fitted on the logit scale. Jumps are pooled over all distinct snapshots.
  interface Fit {
    slope: number;
    residSd: number;
    n: number;
    yLast: number;
    xLast: number;
  }
  const fits: Fit[] = [];
  const allJumps: number[] = [];
  const allResid: number[] = [];
  for (let i = 0; i < nT; i++) {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let j = 0; j < winSnaps.length; j++) {
      const v = winSnaps[j].rates[i];
      if (v !== null && v !== undefined) {
        xs.push(winX[j]);
        ys.push(logit(v));
      }
    }
    // Logit jumps between consecutive distinct snapshots: the refresh volatility.
    for (let j = 1; j < ys.length; j++) allJumps.push(ys[j] - ys[j - 1]);
    const f = fitLine(xs, ys);
    if (f.residSd > 0) allResid.push(f.residSd);
    const lastIdx = ys.length - 1;
    fits.push({
      slope: f.slope,
      residSd: f.residSd,
      n: ys.length,
      yLast: lastIdx >= 0 ? ys[lastIdx] : logit(1),
      xLast: lastIdx >= 0 ? xs[lastIdx] : snapX[snapX.length - 1] ?? 0,
    });
  }

  // Tier-group median slopes (fallback: the global median slope) for shrinkage.
  const globalSlope = median(fits.map((f) => f.slope));
  const tierSlopes = new Map<string, number>();
  for (const tier of new Set(c.trophies.map((t) => t.tier))) {
    const ss = fits.filter((_, i) => c.trophies[i].tier === tier).map((f) => f.slope);
    tierSlopes.set(tier, ss.length > 0 ? median(ss) : globalSlope);
  }

  // Pooled refresh-jump volatility: robust (median absolute jump) with a prior floor.
  const jumpSd = Math.min(
    JUMP_CAP,
    Math.max(JUMP_PRIOR_SD, allJumps.length > 0 ? 1.4826 * median(allJumps.map((d) => Math.abs(d))) : 0),
  );
  const pooledResid = allResid.length > 0 ? median(allResid) : RESID_FLOOR;

  // Median gap between distinct snapshots, in hours: the refresh cadence.
  const gaps: number[] = [];
  for (let j = 1; j < snaps.length; j++) gaps.push(hoursBetween(snaps[j - 1].at, snaps[j].at));
  const refreshH = gaps.length > 0 ? Math.max(0.25, median(gaps)) : REFRESH_INTERVAL_PRIOR_H;
  const horizonH = c.horizon_minutes / 60;
  const expectedRefreshes = Math.min(REFRESH_COUNT_CAP, Math.max(0, horizonH / refreshH));

  const trophies = c.trophies.map((t, i) => {
    const f = fits[i];
    // Shrink the slope toward the tier-group median; anchor the line on the last distinct read.
    const prior = tierSlopes.get(t.tier) ?? globalSlope;
    const w = f.n / (f.n + SLOPE_PRIOR_N);
    const slope = Math.min(SLOPE_MAX, Math.max(SLOPE_MIN, w * f.slope + (1 - w) * prior));
    const dx = Math.max(0, xTarget - f.xLast);
    // Damp and cap the extrapolated move: a saturating rate does not travel as far as its
    // recent local slope suggests over a multi-day horizon.
    const move = Math.min(MOVE_CAP, Math.max(-MOVE_CAP, slope * dx * MOVE_DAMP));
    const yMed = f.yLast + move;

    // Predictive spread, in logit units, combined in quadrature.
    const resid = Math.max(RESID_FLOOR, (f.n * f.residSd + 2 * pooledResid) / (f.n + 2));
    const spread = Math.max(
      SPREAD_FLOOR,
      Math.sqrt(
        resid * resid +
          (SLOPE_PRIOR_SD * dx) ** 2 +
          (jumpSd ** 2) * expectedRefreshes +
          SPREAD_BASE ** 2,
      ),
    );

    const q = Z.map((z) => round4(Math.min(99.99, Math.max(0.01, 100 / (1 + Math.exp(-(yMed + z * spread)))))));
    // Guard the rounding: keep the row non-decreasing.
    for (let k = 1; k < q.length; k++) if (q[k] < q[k - 1]) q[k] = q[k - 1];
    return { id: t.id, q };
  });
  return { trophies };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const forecast = forecastCase(c, file);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: forecast.trophies })}\n`);
}
