"""
Golf Tournament Simulator
=========================
Incorporates all modelled components:
  - Sample size regression to mean (hyperbolic shrinkage)
  - Window deviation (career avg + β × recent deviation)
  - OLS point estimate per metric
  - Skill-dependent skew-normal noise distribution
  - Course-specific variance adjustment
  - Player-specific variance adjustment (ExOtt only)
  - Debut priors for players with no history
  - Cut logic (rank-based, cumulative score, ties survive)

Player names stored in raw format ("Last, First") internally for data
matching, but all output DataFrames and CSVs use clean names ("First Last").

Output DataFrames:
  - round_df       : Round 1 scores  (simulations × players)
  - tournament_df  : Full tournament (simulations × players)
                     Survivors: integer total score
                     Cut players: string "{score}c"
"""

import pandas as pd
import numpy as np
from scipy.stats import skewnorm
from scipy import stats as sp
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ───────────────────────────────────────────────────────────────────
FIELD_CSV        = 'PGAField.csv'
SG_DATA_PATH     = 'SGData.csv'
TOURNAMENT_DATE  = '2026-06-27'
COURSE_NAME      = "TPC River Highlands"
COURSE_PAR       = 70
CUT_AFTER_ROUND  = None     # None = no cut
CUT_NUMBER       = 60       # top N players survive (ties on bubble all survive)
N_SIMULATIONS    = 100_000
N_ROUNDS         = 4
RANDOM_SEED      = 43
MATCHUP_CSV      = 'odds_combined.csv'    # path to matchup csv, or None to skip
# ─────────────────────────────────────────────────────────────────────────────

# ── MODEL PARAMETERS ─────────────────────────────────────────────────────────
PARAMS = {
    'sg_putt': {
        'grand_mean': -0.1103,
        'A': 0.8591, 'k': 12.62,
        'window': 26, 'beta': 0.1802,
        'intercept': -0.0313, 'slope': 0.7285,
        'skew_c': -1.5133, 'skew_d': 0.5010,
        'sigma_a': 1.3838, 'sigma_b': -0.2973,
        'player_variance': False,
    },
    'sg_app_arg': {
        'grand_mean': -0.1745,
        'A': 1.0, 'k': 6.05,
        'window': 40, 'beta': 0.3818,
        'intercept': -0.0258, 'slope': 0.9498,
        'skew_c': -1.9020, 'skew_d': 0.5532,
        'sigma_a': 1.2414, 'sigma_b': -0.2493,
        'player_variance': True,
    },
    'sg_ott': {
        'grand_mean': -0.0367,
        'A': 1.0648, 'k': 5.63,
        'window': 12, 'beta': 0.4810,
        'intercept': -0.0265, 'slope': 0.8611,
        'skew_c': -2.4550, 'skew_d': 0.7807,
        'sigma_a': 1.0424, 'sigma_b': -0.3829,
        'player_variance': False,
    },
}

DEBUT_PRIORS = {
    'sg_ott':  {'career_avg': -0.161, 'pseudo_n': 3},
    'sg_putt': {'career_avg': -0.234, 'pseudo_n': 5},
    'sg_app_arg': {'career_avg': -0.1745, 'pseudo_n': 5},
}

COURSE_SIGMA = {
    'sg_ott': {
        'East Lake Golf Club':                              0.404 / 0.714,
        'Caves Valley Golf Club':                           0.419 / 0.714,
        'Wilmington Country Club':                          0.421 / 0.714,
        'Pebble Beach Golf Links':                          0.900 / 0.714,
        'TPC Sawgrass (THE PLAYERS Stadium Course)':        0.900 / 0.714,
        'Muirfield Village Golf Club':                      1.050 / 0.714,
        'Sea Island Golf Club (Seaside)':                   1.157 / 0.714,
        'Pete Dye Stadium Course':                          0.989 / 0.714,
    },
    'sg_putt': {
        'East Lake Golf Club':                              0.749 / 1.049,
        'Pebble Beach Golf Links':                          1.350 / 1.049,
        'Pete Dye Stadium Course':                          1.471 / 1.049,
        'Sea Island Golf Club (Seaside)':                   1.579 / 1.049,
        'St. Andrews Links (Old Course)':                   1.434 / 1.049,
    },
    'sg_app_arg': {
        'Monterey Peninsula CC':                            2.106 / 1.326,
        'Nicklaus Tournament Course':                       1.824 / 1.326,
        'La Quinta Country Club':                           1.774 / 1.326,
        'Pete Dye Stadium Course':                          1.748 / 1.326,
        'Oakmont Country Club':                             1.715 / 1.326,
        'Torrey Pines Golf Course (North Course)':          1.709 / 1.326,
        'The Philadelphia Cricket Club (Wissahickon Course)': 0.809 / 1.326,
        'Plantation Course at Kapalua':                     0.878 / 1.326,
        'Wilmington Country Club':                          0.880 / 1.326,
        'East Lake Golf Club':                              0.974 / 1.326,
    },
}

# ── NAME UTILITIES ────────────────────────────────────────────────────────────

def clean_name(raw_name):
    """
    Convert 'Last, First' → 'First Last'.
    Names without a comma are returned unchanged.
    """
    raw = str(raw_name).strip()
    if ',' in raw:
        last, first = raw.split(',', 1)
        return f"{first.strip()} {last.strip()}"
    return raw


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_data(sg_path, tournament_date):
    """Load and filter SG data to strictly before tournament date."""
    df = pd.read_csv(sg_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'] < pd.to_datetime(tournament_date)].copy()
    df = df.sort_values(['player_name', 'Date']).reset_index(drop=True)

    # Compute summary SG columns from per-round columns if not already present
    for metric in ['sg_ott', 'sg_putt', 'sg_total']:
        if metric not in df.columns:
            round_cols = [f'round_{r}_{metric}' for r in [1,2,3,4]]
            existing   = [c for c in round_cols if c in df.columns]
            if existing:
                df[metric] = df[existing].mean(axis=1)

    for col in ['sg_ott', 'sg_putt', 'sg_total']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute sg_app_arg = sg_total - sg_putt - sg_ott
    df['sg_app_arg'] = df['sg_total'] - df['sg_putt'] - df['sg_ott']
    return df


def load_field(field_csv):
    """
    Load field CSV, handling both quoted and unquoted 'Last, First' names.

    When names like 'Bhatia, Akshay' appear unquoted in a CSV, pandas splits
    them across two columns. This function reads the file as raw text and
    reconstructs the full name regardless of quoting style.

    Returns:
      raw_names   : list of 'Last, First' names (for SGData matching)
      clean_names : list of 'First Last' names (for output columns)
      raw_to_clean: dict mapping raw → clean
    """
    with open(field_csv, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # Skip header row (first line)
    data_lines = lines[1:]

    raw_names = []
    for line in data_lines:
        if line.startswith('"') and ',' in line:
            # Quoted: "Last, First" — strip outer quotes
            raw_names.append(line.strip('"'))
        elif ',' in line:
            # Unquoted: Last, First (possibly split by CSV parser)
            # Rejoin the two parts cleanly
            parts = [p.strip() for p in line.split(',', 1)]
            raw_names.append(f"{parts[0]}, {parts[1]}")
        else:
            # No comma — use as-is (e.g. single-name or already clean)
            raw_names.append(line.strip())

    clean_names = [clean_name(p) for p in raw_names]
    raw_to_clean = dict(zip(raw_names, clean_names))
    return raw_names, clean_names, raw_to_clean


# ── PLAYER INPUT COMPUTATION ──────────────────────────────────────────────────

def compute_player_inputs(df, raw_names, course_name):
    """
    For each player (matched by raw name) and metric, compute model inputs.
    Returns dict keyed by raw_name.
    """
    field_exott_stds = [
        np.std(df[df['player_name']==p]['sg_app_arg'].dropna().values)
        for p in df['player_name'].unique()
        if len(df[df['player_name']==p]['sg_app_arg'].dropna()) >= 10
    ]
    field_mean_std = np.mean(field_exott_stds) if field_exott_stds else 1.0

    player_inputs = {}
    for player in raw_names:
        pdata = df[df['player_name'] == player].sort_values('Date')
        player_inputs[player] = {}

        for metric in ['sg_ott', 'sg_putt', 'sg_app_arg']:
            p    = PARAMS[metric]
            vals = pdata[metric].dropna().values

            if len(vals) == 0:
                career_avg = DEBUT_PRIORS[metric]['career_avg']
                n_rounds   = DEBUT_PRIORS[metric]['pseudo_n']
                window_avg = career_avg
            else:
                career_avg = float(np.mean(vals))
                n_rounds   = len(vals)
                w          = p['window']
                window_avg = float(np.mean(vals[-w:])) if n_rounds >= w else career_avg

            player_sigma_ratio = 1.0
            if p['player_variance'] and len(vals) >= 10:
                player_std = float(np.std(vals))
                player_sigma_ratio = np.clip(
                    player_std / field_mean_std if field_mean_std > 0 else 1.0,
                    0.5, 2.5
                )

            course_sigma_ratio = COURSE_SIGMA.get(metric, {}).get(course_name, 1.0)

            player_inputs[player][metric] = {
                'career_avg':         career_avg,
                'window_avg':         window_avg,
                'n_rounds':           n_rounds,
                'player_sigma_ratio': player_sigma_ratio,
                'course_sigma_ratio': course_sigma_ratio,
            }

    return player_inputs


# ── POINT ESTIMATE ────────────────────────────────────────────────────────────

def point_estimate(metric, career_avg, window_avg, n_rounds):
    p     = PARAMS[metric]
    trust = p['A'] * n_rounds / (n_rounds + p['k'])
    ca_shrunk = p['grand_mean'] + trust * (career_avg - p['grand_mean'])
    if n_rounds >= p['window']:
        predictor = ca_shrunk + p['beta'] * (window_avg - career_avg) * trust
    else:
        predictor = ca_shrunk
    return float(p['intercept'] + p['slope'] * predictor)


# ── SIMULATION DRAW ───────────────────────────────────────────────────────────

def simulate_metric(rng, metric, skill, player_sigma_ratio, course_sigma_ratio, n_sims):
    p     = PARAMS[metric]
    alpha = float(np.clip(p['skew_c'] + p['skew_d'] * skill, -10.0, -0.05))
    sigma = float(np.clip(p['sigma_a'] + p['sigma_b'] * skill,  0.20,  5.00))
    sigma *= course_sigma_ratio * player_sigma_ratio
    delta = alpha / np.sqrt(1.0 + alpha**2)
    loc   = skill - sigma * np.sqrt(2.0 / np.pi) * delta
    return skewnorm.rvs(a=alpha, loc=loc, scale=sigma, size=n_sims, random_state=rng)


# ── ROUND SIMULATION ──────────────────────────────────────────────────────────

def simulate_round(rng, raw_names, clean_names, player_inputs, par, n_sims):
    """
    Simulate one round. Returns DataFrame with clean names as columns.
    """
    scores = {}
    for raw, clean in zip(raw_names, clean_names):
        total_sg = np.zeros(n_sims)
        for metric in ['sg_ott', 'sg_putt', 'sg_app_arg']:
            inp   = player_inputs[raw][metric]
            skill = point_estimate(
                metric, inp['career_avg'], inp['window_avg'], inp['n_rounds']
            )
            total_sg += simulate_metric(
                rng, metric, skill,
                inp['player_sigma_ratio'],
                inp['course_sigma_ratio'],
                n_sims,
            )
        scores[clean] = np.round(par - total_sg).astype(int)

    return pd.DataFrame(scores)


# ── MAIN SIMULATION LOOP ──────────────────────────────────────────────────────

def run_tournament(
    field_csv       = FIELD_CSV,
    sg_data_path    = SG_DATA_PATH,
    tournament_date = TOURNAMENT_DATE,
    course_name     = COURSE_NAME,
    course_par      = COURSE_PAR,
    cut_after_round = CUT_AFTER_ROUND,
    cut_number      = CUT_NUMBER,
    n_rounds        = N_ROUNDS,
    n_sims          = N_SIMULATIONS,
    seed            = RANDOM_SEED,
    matchup_csv     = MATCHUP_CSV,
):
    print("Loading data...")
    df = load_data(sg_data_path, tournament_date)
    raw_names, clean_names, raw_to_clean = load_field(field_csv)
    print(f"  Field: {len(raw_names)} players")
    print(f"  SG data: {len(df)} rows before {tournament_date}")

    print("Computing player inputs...")
    player_inputs = compute_player_inputs(df, raw_names, course_name)

    # Historical rounds per player (sg_ott as representative metric)
    historical_rounds = {}
    for raw, clean in zip(raw_names, clean_names):
        vals = df[df['player_name'] == raw]['sg_ott'].dropna()
        historical_rounds[clean] = len(vals)

    # ── Players with no historical data ───────────────────────────────────────
    no_history_raw   = [p for p in raw_names
                        if all(df[df['player_name']==p][m].dropna().empty
                               for m in ['sg_ott','sg_putt','sg_app_arg'])]
    no_history_clean = [raw_to_clean[p] for p in no_history_raw]

    if no_history_clean:
        print(f"\n{'='*55}")
        print(f"WARNING: {len(no_history_clean)} player(s) had no historical data.")
        print("Debut priors used for:")
        for name in no_history_clean:
            print(f"  - {name}")
        print(f"  sg_ott  prior: {DEBUT_PRIORS['sg_ott']['career_avg']}  "
              f"(pseudo_n={DEBUT_PRIORS['sg_ott']['pseudo_n']})")
        print(f"  sg_putt prior: {DEBUT_PRIORS['sg_putt']['career_avg']}  "
              f"(pseudo_n={DEBUT_PRIORS['sg_putt']['pseudo_n']})")
        print(f"  sg_app_arg prior: {DEBUT_PRIORS['sg_app_arg']['career_avg']}  "
              f"(pseudo_n={DEBUT_PRIORS['sg_app_arg']['pseudo_n']})")
        print('='*55 + '\n')
    else:
        print("  All players had historical data — no debut priors used.")

    rng = np.random.default_rng(seed)

    # ── Running totals and state — indexed by clean names ────────────────────
    cumulative = pd.DataFrame(
        np.zeros((n_sims, len(clean_names)), dtype=int),
        columns=clean_names,
    )
    active = pd.DataFrame(
        np.ones((n_sims, len(clean_names)), dtype=bool),
        columns=clean_names,
    )
    tournament_df = pd.DataFrame(index=range(n_sims), columns=clean_names, dtype=object)
    round_df      = None

    for round_num in range(1, n_rounds + 1):
        print(f"Simulating round {round_num}...")

        round_scores = simulate_round(
            rng, raw_names, clean_names, player_inputs, course_par, n_sims
        )

        # Inactive players don't accumulate score
        cumulative += round_scores.where(active, other=0)

        if round_num == 1:
            round_df = round_scores.copy()
            round_df.index.name = 'simulation'

        # ── Apply cut ─────────────────────────────────────────────────────────
        if cut_after_round is not None and round_num == cut_after_round:
            print(f"  Applying cut (top {cut_number} survive, ties on bubble kept)...")

            # Vectorised rank across all sims at once
            # method='min': tied players all get lowest rank → ties survive
            ranks = cumulative.rank(axis=1, method='min', ascending=True)
            cut_mask = ranks > cut_number   # shape: (n_sims, n_players)

            # Update active — cut players become inactive
            active = active & ~cut_mask

            # Write cut-player scores (with 'c') — vectorised
            # Done after all rounds complete (see below)

            n_cut = cut_mask.sum(axis=1).mean()
            print(f"  Average players cut per simulation: {n_cut:.1f}")

    # ── Build tournament_df vectorised ───────────────────────────────────────
    # Survivors get their integer cumulative score.
    # Cut players get their score at cut time as a string with 'c' suffix.
    # active = True for survivors (final state after all cuts applied)
    cut_final = ~active   # True where player was cut

    # Start with cumulative scores as strings, append 'c' for cut players
    scores_str = cumulative.astype(str)
    scores_str[cut_final] = scores_str[cut_final] + 'c'

    # Survivors: use integer values; cut: use string with 'c'
    tournament_df = scores_str.where(cut_final, cumulative.astype(object))
    tournament_df.index.name = 'simulation'

    print(f"\nDone.")
    print(f"  round_df shape:      {round_df.shape}")
    print(f"  tournament_df shape: {tournament_df.shape}")

    # ── Placement percentages ────────────────────────────────────────────────
    # Cut players (identified by 'c' suffix) are ineligible for any placement.
    # Ties are handled fractionally: if N players tie for a position that
    # straddles a threshold (e.g. 3-way tie for 4th when threshold=5),
    # each tied player receives a fractional credit proportional to how many
    # of the tied group fall within the threshold.
    #
    # Example: 4-way tie for 4th, threshold=5 → 2 of 4 slots remain
    #          → each of the 4 tied players gets 2/4 = 0.5 credit

    # Build numeric scores (NaN for cut players)
    numeric_scores = pd.DataFrame(index=range(n_sims), columns=clean_names, dtype=float)
    for clean in clean_names:
        for sim_idx in range(n_sims):
            val = tournament_df.at[sim_idx, clean]
            if not str(val).endswith('c'):
                numeric_scores.at[sim_idx, clean] = float(val)

    def fractional_top_n(scores_df, n):
        """
        Vectorised fractional top-N calculation across all simulations at once.
        Cut players (NaN) are replaced with inf so they sort to the end.
        Ties straddling position N receive fractional credit:
            credit = slots_within_top_N / tie_group_size
        """
        arr     = scores_df.values.copy().astype(float)
        n_sims_l, n_players_l = arr.shape
        nan_mask   = np.isnan(arr)
        arr_filled = np.where(nan_mask, np.inf, arr)
        sort_idx   = np.argsort(arr_filled, axis=1, kind='stable')
        sorted_arr = np.take_along_axis(arr_filled, sort_idx, axis=1)
        credits    = np.zeros((n_sims_l, n_players_l), dtype=float)
        sim_idx_all = np.arange(n_sims_l)

        for j in range(min(n, n_players_l)):
            score_j = sorted_arr[:, j]
            valid   = ~np.isinf(score_j)
            if not valid.any():
                break

            # Only process each tie group at its first occurrence
            if j == 0:
                is_first = valid
            else:
                is_first = valid & (score_j != sorted_arr[:, j - 1])

            if not is_first.any():
                continue

            # Find where the tie group ends (scan forward)
            tie_end = np.full(n_sims_l, j, dtype=int)
            for k in range(j + 1, n_players_l):
                same = (sorted_arr[:, k] == score_j) & valid
                tie_end = np.where(same & is_first, k, tie_end)

            tie_size      = tie_end - j + 1
            slots_in_topn = np.maximum(0, np.minimum(tie_end + 1, n) - j)
            frac          = np.where(is_first & (tie_size > 0),
                                     slots_in_topn / tie_size, 0.0)

            # Assign frac to all players in each tie group
            for k in range(j, n_players_l):
                if k >= n:
                    # Beyond top-N — only assign if tie group started before N
                    in_group = is_first & (k <= tie_end) & (sorted_arr[:, k] == score_j) & valid
                else:
                    in_group = is_first & (k <= tie_end) & (sorted_arr[:, k] == score_j) & valid
                if not in_group.any():
                    break
                at_k = sort_idx[:, k]
                credits[sim_idx_all[in_group], at_k[in_group]] = frac[in_group]

        credits[nan_mask] = 0.0
        return pd.DataFrame(credits, columns=scores_df.columns)

    # Win (top 1)
    sim_min   = numeric_scores.min(axis=1)
    is_winner = numeric_scores.eq(sim_min, axis=0)
    n_winners = is_winner.sum(axis=1)
    frac_win  = is_winner.div(n_winners, axis=0).fillna(0.0)
    win_pct   = (frac_win.sum(axis=0) / n_sims * 100).round(4)

    # Top 5
    frac_top5  = fractional_top_n(numeric_scores, 5)
    top5_pct   = (frac_top5.sum(axis=0) / n_sims * 100).round(4)

    # Top 10
    frac_top10 = fractional_top_n(numeric_scores, 10)
    top10_pct  = (frac_top10.sum(axis=0) / n_sims * 100).round(4)

    place_df = pd.DataFrame({
        'player':           clean_names,
        'historical_rounds':[historical_rounds[p] for p in clean_names],
        'win_pct':          [win_pct[p]  for p in clean_names],
        'top5_pct':         [top5_pct[p] for p in clean_names],
        'top10_pct':        [top10_pct[p] for p in clean_names],
    }).sort_values('win_pct', ascending=False).reset_index(drop=True)
    place_df.index = place_df.index + 1
    place_df.index.name = 'rank'

    # ── Made the cut % ───────────────────────────────────────────────────────
    # A player made the cut if their tournament_df value does NOT end with 'c'.
    # If there is no cut, all active players are considered to have made it.
    if cut_after_round is not None:
        # active is a bool DataFrame — sum survivors per column, vectorised
        made_cut_vals = (active.sum(axis=0) / n_sims * 100).round(4)
        cut_df = pd.DataFrame({
            'player':           clean_names,
            'historical_rounds':[historical_rounds[p] for p in clean_names],
            'made_cut_pct':     [made_cut_vals[p] for p in clean_names],
        }).sort_values('made_cut_pct', ascending=False).reset_index(drop=True)
        cut_df.index = cut_df.index + 1
        cut_df.index.name = 'rank'
    else:
        cut_df = pd.DataFrame({
            'player':           clean_names,
            'historical_rounds':[historical_rounds[p] for p in clean_names],
            'made_cut_pct':     [100.0] * len(clean_names),
        }).sort_values('made_cut_pct', ascending=False).reset_index(drop=True)
        cut_df.index = cut_df.index + 1
        cut_df.index.name = 'rank'

    # ── Matchup win percentages ──────────────────────────────────────────────
    matchup_df = None
    if matchup_csv is not None:
        import openpyxl
        mu_raw = pd.read_csv(matchup_csv)
        # Support both formats:
        #   odds_combined.csv  — columns: source, player_1, player_2, ...
        #   simple matchup csv — columns: Player 1, Player 2  (or first two cols)
        if 'player_1' in mu_raw.columns and 'player_2' in mu_raw.columns:
            p1_col, p2_col = 'player_1', 'player_2'
        elif 'Player 1' in mu_raw.columns and 'Player 2' in mu_raw.columns:
            p1_col, p2_col = 'Player 1', 'Player 2'
        else:
            # Fall back to first two columns
            p1_col = mu_raw.columns[0]
            p2_col = mu_raw.columns[1]

        # Build a lookup: clean_name → clean_name (identity, for validation)
        clean_set = set(clean_names)

        # For each simulation, get the final score for each player.
        # Use numeric_scores (NaN = cut) for eligibility, but for
        # cut-vs-cut matchups fall back to the raw score at cut time.
        # Build a raw numeric matrix including cut scores (no NaN).
        raw_numeric = pd.DataFrame(index=range(n_sims), columns=clean_names, dtype=float)
        for clean in clean_names:
            for sim_idx in range(n_sims):
                val = tournament_df.at[sim_idx, clean]
                raw_numeric.at[sim_idx, clean] = float(str(val).replace('c', ''))

        # Detect optional columns from odds_combined.csv
        has_source  = 'source'  in mu_raw.columns
        has_odds    = 'odds_1'  in mu_raw.columns and 'odds_2' in mu_raw.columns

        matchup_rows = []
        for _, row in mu_raw.iterrows():
            p1_raw = str(row[p1_col]).strip()
            p2_raw = str(row[p2_col]).strip()
            p1 = clean_name(p1_raw)
            p2 = clean_name(p2_raw)
            book   = str(row['source'])  if has_source else None
            odds_1 = int(row['odds_1'])  if has_odds   else None
            odds_2 = int(row['odds_2'])  if has_odds   else None

            # Validate both players are in the field
            p1_in = p1 in clean_set
            p2_in = p2 in clean_set

            if not p1_in or not p2_in:
                missing = []
                if not p1_in: missing.append(p1)
                if not p2_in: missing.append(p2)
                print(f"  WARNING: matchup skipped — not in field: {missing}")
                matchup_rows.append({
                    'book':                  book,
                    'player_1':              p1,
                    'player_2':              p2,
                    'odds_1':                odds_1,
                    'odds_2':                odds_2,
                    'p1_historical_rounds':  historical_rounds.get(p1, 0),
                    'p2_historical_rounds':  historical_rounds.get(p2, 0),
                    'p1_win_pct':            None,
                    'p2_win_pct':            None,
                    'tie_pct':               None,
                })
                continue

            p1_scores = raw_numeric[p1].values
            p2_scores = raw_numeric[p2].values

            # Determine winner per simulation:
            # - If both cut: use their cut score (lower wins)
            # - If one cut, other not: non-cut player wins
            # - If neither cut: lower final score wins
            # - If tied: ignored (neither wins)
            p1_cut = np.array([str(tournament_df.at[i, p1]).endswith('c')
                                for i in range(n_sims)])
            p2_cut = np.array([str(tournament_df.at[i, p2]).endswith('c')
                                for i in range(n_sims)])

            both_cut   = p1_cut & p2_cut
            neither_cut= ~p1_cut & ~p2_cut
            p1_only_cut= p1_cut & ~p2_cut
            p2_only_cut= ~p1_cut & p2_cut

            p1_wins = np.zeros(n_sims, dtype=bool)
            p2_wins = np.zeros(n_sims, dtype=bool)
            tied    = np.zeros(n_sims, dtype=bool)

            # Both cut — use cut score
            p1_wins |= both_cut & (p1_scores < p2_scores)
            p2_wins |= both_cut & (p2_scores < p1_scores)
            tied    |= both_cut & (p1_scores == p2_scores)

            # Neither cut — use final score
            p1_wins |= neither_cut & (p1_scores < p2_scores)
            p2_wins |= neither_cut & (p2_scores < p1_scores)
            tied    |= neither_cut & (p1_scores == p2_scores)

            # One cut — non-cut player wins
            p1_wins |= p2_only_cut
            p2_wins |= p1_only_cut

            n_decided = n_sims - tied.sum()
            p1_pct = round(p1_wins.sum() / n_sims * 100, 4)
            p2_pct = round(p2_wins.sum() / n_sims * 100, 4)
            tie_pct = round(tied.sum() / n_sims * 100, 4)

            # Net % excludes ties — p1_net + p2_net = 100
            n_decided = n_sims - tied.sum()
            if n_decided > 0:
                p1_net = round(p1_wins.sum() / n_decided * 100, 4)
                p2_net = round(p2_wins.sum() / n_decided * 100, 4)
            else:
                p1_net = p2_net = 50.0

            # Ties lose books: report raw win% and tie% (all three sum to 100)
            # Ties void books: report net% only (p1+p2=100), tie_pct=None
            ties_lose = (book in {'Bet365'})
            matchup_rows.append({
                'book':                 book,
                'player_1':             p1,
                'player_2':             p2,
                'odds_1':               odds_1,
                'odds_2':               odds_2,
                'p1_historical_rounds': historical_rounds.get(p1, 0),
                'p2_historical_rounds': historical_rounds.get(p2, 0),
                'p1_win_pct':           round(p1_wins.sum() / n_sims * 100, 4) if ties_lose else p1_net,
                'p2_win_pct':           round(p2_wins.sum() / n_sims * 100, 4) if ties_lose else p2_net,
                'tie_pct':              tie_pct if ties_lose else None,
            })

        matchup_df = pd.DataFrame(matchup_rows)
        matchup_df.index = matchup_df.index + 1
        matchup_df.index.name = 'matchup'
    # ── Player stats (projected values + SDs) ───────────────────────────────
    stats_rows = []
    for raw, clean in zip(raw_names, clean_names):
        row = {'player': clean, 'historical_rounds': historical_rounds[clean]}
        total_proj = 0.0
        total_sd2  = 0.0
        for metric in ['sg_ott', 'sg_putt', 'sg_app_arg']:
            inp   = player_inputs[raw][metric]
            skill = point_estimate(
                metric, inp['career_avg'], inp['window_avg'], inp['n_rounds']
            )
            p     = PARAMS[metric]
            sigma = float(np.clip(p['sigma_a'] + p['sigma_b'] * skill, 0.20, 5.00))
            sigma *= inp['course_sigma_ratio'] * inp['player_sigma_ratio']
            row[f'{metric}_proj'] = round(skill, 4)
            row[f'{metric}_sd']   = round(sigma, 4)
            total_proj += skill
            total_sd2  += sigma ** 2   # variances are additive (assumed independent)
        row['total_sg_proj'] = round(total_proj, 4)
        row['total_sg_sd']   = round(float(np.sqrt(total_sd2)), 4)
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows).sort_values('total_sg_proj', ascending=False).reset_index(drop=True)
    stats_df.index = stats_df.index + 1
    stats_df.index.name = 'rank'

    # ── Merge cut % into place_df and add place odds ─────────────────────────
    # Add miss_cut_pct to place_df from cut_df
    place_df = place_df.merge(
        cut_df[['player','made_cut_pct']].rename(columns={'made_cut_pct':'make_cut_pct'}),
        on='player', how='left'
    )
    place_df['miss_cut_pct'] = (100 - place_df['make_cut_pct']).round(4)

    # Merge book odds if combined_place_odds.csv exists
    try:
        place_odds = pd.read_csv('combined_place_odds.csv')
        # Only keep book columns that exist in the file
        wanted = ['player','win_book','top5_book','top10_book',
                  'make_cut_book','miss_cut_book']
        available = [c for c in wanted if c in place_odds.columns]
        if len(available) > 1:   # at least 'player' + one odds column
            place_df = place_df.merge(place_odds[available], on='player', how='left')
    except FileNotFoundError:
        pass

    # Final column order for place_df
    place_cols = ['player','historical_rounds',
                  'win_pct','top5_pct','top10_pct','make_cut_pct','miss_cut_pct',
                  'win_book','top5_book','top10_book','make_cut_book','miss_cut_book']
    place_df = place_df[[c for c in place_cols if c in place_df.columns]]

    # ── Write CSVs ────────────────────────────────────────────────────────────
    round_df.to_csv('round_scores.csv')
    tournament_df.to_csv('tournament_scores.csv')
    place_df.to_csv('place.csv')
    if matchup_df is not None:
        matchup_df.to_csv('matchups.csv')
    stats_df.to_csv('stats.csv')

    print(f"\nCSV output written:")
    print(f"  Round 1 scores    : round_scores.csv")
    print(f"  Tournament scores : tournament_scores.csv")
    print(f"  Win percentages   : place.csv")
    if matchup_df is not None:
        print(f"  Matchups          : matchups.csv")
    print(f"  Player stats      : stats.csv")

    print(f"\nTop 10 placement probabilities:")
    print(place_df.head(10).to_string())


    if matchup_df is not None:
        print(f"\nMatchup results:")
        print(matchup_df.to_string())

    print(f"\nPlayer stats:")
    print(stats_df.to_string())

    return round_df, tournament_df, place_df, cut_df, matchup_df, stats_df


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    round_df, tournament_df, place_df, cut_df, matchup_df, stats_df = run_tournament()
