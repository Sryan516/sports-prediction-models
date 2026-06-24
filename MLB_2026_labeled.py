# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 21:53:15 2020

@author: Sean Ryan

MLB Game Simulator
------------------
Monte Carlo simulation of MLB games. For each matchup, the model runs 1,000
simulated games by stepping through each at-bat using probabilistic outcomes
derived from pitcher stats, batter stats, park factors, weather, and
situational adjustments. Outputs a CSV of simulated box score results per game.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import random as rand


# ---------------------------------------------------------------------------
# Situational adjustment multipliers
# ---------------------------------------------------------------------------
# Strikeout rate adjustments indexed by [situation (0=empty, 1=partial, 2=loaded), outs (0,1,2)]
# Each value scales the base SO probability up or down based on base/out state
SituationStrikeout = [0.997643259,  # empty, 0 outs
                      0.895739753,  # partial, 0 outs
                      0.914693594,  # loaded, 0 outs
                      1.04617148,   # empty, 1 out
                      0.936319337,  # partial, 1 out
                      0.946264507,  # loaded, 1 out
                      1.086330642,  # empty, 2 outs
                      1.0015385,    # partial, 2 outs
                      1.009356119]  # loaded, 2 outs

# Walk rate adjustments indexed by same situation/out scheme
SituationBB = [0.915389876,
               0.946425197,
               0.769812296,
               0.936147738,
               1.040113722,
               0.819468013,
               1.004172325,
               1.171563165,
               1.021231318]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
directory = 'C:\\Noonedrive\\'
outdir    = 'C:\\Noonedrive\\MLBGameFiles\\'

# Batter and pitcher statistical tables (expected rates per plate appearance)
battersdata  = pd.read_csv(directory + 'Battersdata.csv',  encoding='cp1252')
pitchersdata = pd.read_csv(directory + 'Pitchersdata.csv', encoding='cp1252')

# Batter handedness lookup (used to determine pull/oppo direction vs. field side)
Handedness = pd.read_csv(directory + 'MLBHandedness.csv', encoding='cp1252')
Handedness = Handedness.drop_duplicates(keep='first')

# Park factors: hit type and direction multipliers per ballpark
PF = pd.read_csv(directory + 'NewMLBParkFactorInSeason.csv')

# Weather data: temperature and wind conditions per home team
Weather = pd.read_csv(directory + 'NewMLBWeather.csv')

# Pitching and batting lineup files (ordered 1-9 per team)
Pitchlineups  = pd.read_csv(directory + 'MLBPitchingLineup.csv',  encoding='cp1252')
Battinglineups = pd.read_csv(directory + 'MLBBattingLineup.csv', encoding='cp1252')

# League average rates used as the base for all probability calculations
Averages = pd.read_csv(directory + 'MLBAverages.csv')


# ---------------------------------------------------------------------------
# Index all lookup tables by their key column for fast .loc[] access
# ---------------------------------------------------------------------------
battersdata  = battersdata.drop_duplicates(keep='first')
battersdata.index  = battersdata['Name']
pitchersdata.index = pitchersdata['Name']

PF.index        = PF['Team']
Weather.index   = Weather['Team']
Averages.index  = Averages['Averages']
Handedness.index = Handedness['NAME']


# ---------------------------------------------------------------------------
# Initialise global state (reset per game in the main loop)
# ---------------------------------------------------------------------------
sumcols = []

# Build placeholder column list for inning scores (A1-A15, H1-H15)
counter = 1
while counter < 16:
    sumcols.append('A' + str(counter))
    counter = counter + 1
counter = 1
while counter < 16:
    sumcols.append('H' + str(counter))
    counter = counter + 1

inning = 1   # current inning tracker (global, referenced inside runoffense)

# Pitch count and batter position counters (home/away starters)
awaycp = 0
homecp = 0
awaycb = 0
homecb = 0


# ---------------------------------------------------------------------------
# Core simulation function: simulate one half-inning of offense
# ---------------------------------------------------------------------------
def runoffense(side, cp, pc, cb):
    """
    Simulate one half-inning for the batting team.

    Parameters
    ----------
    side : str
        'away' or 'home' — which team is batting
    cp   : int
        Current pitcher's cumulative batter count (used to determine starter vs. reliever)
    pc   : int
        Starter's expected batters faced (BF) before being pulled
    cb   : int
        Current batter index in the lineup (0-8)

    Returns
    -------
    (score, cp, cb) : tuple
        Runs scored, updated pitcher count, updated batter index
    """

    # Initialise base state and counters for this half-inning
    outs    = 0
    first   = 0   # runner on first  (0 = empty, else player name)
    second  = 0   # runner on second
    third   = 0   # runner on third
    score   = 0
    counter = 0

    # Extra innings rule: place runner on second at start of inning
    if side == 'home':
        if inning > 9:
            second = HomeBatters[8] if cb == 0 else HomeBatters[cb - 1]
    else:
        if inning > 9:
            second = Awaybatters[8] if cb == 0 else Awaybatters[cb - 1]

    # ------------------------------------------------------------------
    # At-bat loop: continue until 3 outs
    # ------------------------------------------------------------------
    while outs < 3:

        # Determine which pitcher is throwing based on pitch count vs. capacity
        if side == 'away':
            std = Away
            if cp <= pc:
                Pitcher = HomePitchers[0]
                inningso = 0.95 if inning > 4 else 1  # starter fatigue factor after inning 4
            else:
                Pitcher  = HomePitchers[inning - 1]   # reliever indexed by inning
                inningso = 1
            Batter = Awaybatters[cb]
        else:
            std = Home
            if cp <= pc:
                Pitcher = AwayPitchers[0]
                inningso = 0.95 if inning > 4 else 1
            else:
                Pitcher  = AwayPitchers[inning - 1]
                inningso = 1
            Batter = HomeBatters[cb]

        # ------------------------------------------------------------------
        # Pull pitcher and batter stats for this matchup
        # ------------------------------------------------------------------
        PSO = pitchersdata.loc[Pitcher]['ESO%']   # pitcher expected SO rate
        PBO = pitchersdata.loc[Pitcher]['EBB%']   # pitcher expected BB rate
        PGB = pitchersdata.loc[Pitcher]['EGB%']   # pitcher expected GB rate
        PFB = pitchersdata.loc[Pitcher]['EFB%']   # pitcher expected FB rate

        BSO = battersdata.loc[Batter]['ESO%']              # batter expected SO rate
        BBB = battersdata.loc[Batter]['EBB%']              # batter expected BB rate
        BFB = battersdata.loc[Batter]['EFB%']              # batter expected FB rate
        BGB = battersdata.loc[Batter]['EGB%']              # batter expected GB rate
        BHH = battersdata.loc[Batter]['AHRD%']             # batter hard-hit rate
        BSH = battersdata.loc[Batter]['ASFT%']             # batter soft-hit rate
        BPH = battersdata.loc[Batter]['EffectiverPull%']   # batter pull tendency
        BOH = battersdata.loc[Batter]['EffectiverOppo%']   # batter oppo tendency

        # ------------------------------------------------------------------
        # Determine base situation (empty / partial / loaded) for adjustments
        # ------------------------------------------------------------------
        if first == 0 and second == 0 and third == 0:
            Situation = 0   # bases empty
        elif first != 0 and second != 0 and third != 0:
            Situation = 2   # bases loaded
        else:
            Situation = 1   # at least one on, not loaded

        # Select the appropriate situational multipliers for SO and BB
        idx = Situation + (outs * 3)
        SituationPreSO = SituationStrikeout[idx]
        SituationPreBB = SituationBB[idx]

        # ------------------------------------------------------------------
        # Loaded/partial base adjustments for hit type probabilities
        # ------------------------------------------------------------------
        if Situation == 2:
            loadedhardadjusted  = 1.038338658
            loadedsoftadjusted  = 0.981
            loadedpulladjusted  = 1.034778325
            loadedoppadjusted   = 0.958677686
        elif Situation == 1:
            loadedhardadjusted  = 1.013194888
            loadedsoftadjusted  = 0.98757764
            loadedpulladjusted  = 1.004926108
            loadedoppadjusted   = 1.004132231
        else:
            loadedhardadjusted  = 1
            loadedsoftadjusted  = 1
            loadedpulladjusted  = 1
            loadedoppadjusted   = 1

        # ------------------------------------------------------------------
        # Calculate at-bat outcome probabilities
        # Home/away splits applied to SO and BB rates
        # Combined: league average * pitcher rate * batter rate * park factor * situation * fatigue
        # ------------------------------------------------------------------
        if side == 'away':
            SO = (1.022869862 * Averages.loc['SO']['Percent'] * PSO * BSO
                  * PF.loc[Home]['StrikeOut'] * SituationPreSO * inningso + TempAdjSo)
        else:
            SO = (0.977130138 * Averages.loc['SO']['Percent'] * PSO * BSO
                  * PF.loc[Home]['StrikeOut'] * SituationPreSO * inningso + TempAdjSo)

        if side == 'away':
            BB = (0.961176604 * Averages.loc['BB']['Percent'] * PBO * BBB
                  * PF.loc[Home]['BB'] * SituationPreBB)
        else:
            BB = (1.038823396 * Averages.loc['BB']['Percent'] * PBO * BBB
                  * PF.loc[Home]['BB'] * SituationPreBB)

        # ------------------------------------------------------------------
        # Roll a random number to determine the at-bat outcome
        # ------------------------------------------------------------------
        event = rand.random()

        if event < SO:
            result = 'K'
        elif event < (SO + BB):
            result = 'walk'

        else:
            # Ball in play: determine trajectory (GB / LD / FB)
            GB = Averages.loc['GB']['Percent'] * PGB * BGB * PF.loc[Home]['GB']
            LD = Averages.loc['LD']['Percent'] * PF.loc[Home]['LD']
            FB = Averages.loc['FB']['Percent'] * PFB * BFB * PF.loc[Home]['FB']

            # Normalise so GB + LD + FB = 1
            Diff  = 1 - LD
            ratio = Diff / (GB + FB)
            Gbs   = GB * ratio
            FBs   = FB * ratio
            Lds   = LD

            traj = rand.random()

            # Determine trajectory bucket and derive direction + magnitude
            if traj < Gbs:
                trj = 'GB'
            elif traj < (Gbs + Lds):
                trj = 'LD'
            else:
                trj = 'FB'

            # Direction: Pull / Center / Oppo (weighted by batter tendency + situation)
            Pull   = Averages.loc['Pull '   + trj]['Percent'] * BPH * loadedpulladjusted
            Center = Averages.loc['Center ' + trj]['Percent']
            Opp    = Averages.loc['Opp '    + trj]['Percent'] * BOH * loadedoppadjusted

            drctscale = Pull + Center + Opp
            Pulls   = Pull   / drctscale
            Centers = Center / drctscale
            Opps    = Opp    / drctscale

            drct = rand.random()
            if drct < Pulls:
                dirct = 'Pull'
            elif drct < (Pulls + Centers):
                dirct = 'Center'
            else:
                dirct = 'Opp'

            # Magnitude: Hard / Medium / Soft
            Hard = (Averages.loc['Hard ' + trj]['Percent'] * BHH
                    * PF.loc[Home]['Hard'] * loadedhardadjusted + TempADjHard)
            Soft = (Averages.loc['Soft ' + trj]['Percent'] * BSH
                    * PF.loc[Home]['Soft'] * loadedsoftadjusted + TempAdjSoft)

            magnitude = rand.random()
            if magnitude < Soft:
                mag = 'Soft'
            elif magnitude > (1 - Hard):
                mag = 'Hard'
            else:
                mag = 'Med'

            # Translate pull direction to field side based on batter handedness
            if Handedness.loc[Batter]['RAT'] == 'R':
                btw = 'Left' if dirct == 'Pull' else ('Right' if dirct == 'Opp' else 'Center')
            else:
                btw = 'Right' if dirct == 'Pull' else ('Left' if dirct == 'Opp' else 'Center')

            # Select directional park factors for each hit type
            if btw == 'Left':
                pf1b = PF.loc[Home]['LSF'];  pf2b = PF.loc[Home]['LDf']
                pf3b = PF.loc[Home]['LTPF']; pfhr = PF.loc[Home]['LHRf']
            elif btw == 'Center':
                pf1b = PF.loc[Home]['CSF'];  pf2b = PF.loc[Home]['Cdf']
                pf3b = PF.loc[Home]['CTF'];  pfhr = PF.loc[Home]['CHRG']
            else:
                pf1b = PF.loc[Home]['RSF'];  pf2b = PF.loc[Home]['RDF']
                pf3b = PF.loc[Home]['RTF'];  pfhr = PF.loc[Home]['RTHRF']

            # Final BIP outcome probabilities
            # Key: trj + dirct + mag identifies a unique cell in the Averages lookup table
            sin = Averages.loc[trj + ' ' + dirct + ' ' + mag + ' Single']['Percent'] * pf1b + TempAdjSin
            db  = Averages.loc[trj + ' ' + dirct + ' ' + mag + ' DB'    ]['Percent'] * pf2b + TempAdjDb
            trp = Averages.loc[trj + ' ' + dirct + ' ' + mag + ' TRP'   ]['Percent'] * pf3b
            HR  = (Averages.loc[trj + ' ' + dirct + ' ' + mag + ' HR'   ]['Percent'] * pfhr
                   + TempAdjHR + WindAdHR)
            out = 1 - HR - db - trp - sin

            # Roll to determine final BIP result
            bipresult = rand.random()
            if bipresult < out:
                result = 'out'
            elif bipresult < (out + sin):
                result = 'Sin'
            elif bipresult < (out + sin + db):
                result = 'db'
            elif bipresult < (out + sin + db + trp):
                result = 'trp'
            else:
                result = 'HR'

        # ------------------------------------------------------------------
        # Baserunning: two random rolls used to resolve runner advancement
        # ------------------------------------------------------------------
        baserun1 = rand.random()   # used for lead runner advancement decisions
        baserun2 = rand.random()   # used for trailing runner scoring decisions

        # ------------------------------------------------------------------
        # Apply result to base state and accumulate stats in summary DataFrame
        # ------------------------------------------------------------------
        if result == 'K':
            outs = outs + 1
            if Pitcher == AwayPitchers[0] or Pitcher == HomePitchers[0]:
                summary.at[0, Pitcher + 'SO'] = summary.iloc[0][Pitcher + 'SO'] + 1

        elif result == 'walk':
            # Advance runners on walk; force score only if bases loaded
            if first == 0:
                first = Batter
            elif first != 0 and second == 0:
                second = first; first = Batter
            elif first != 0 and second != 0 and third == 0:
                third = second; second = first; first = Batter
            else:
                score = score + 1
                summary.at[0, Batter + 'RBI']   = summary.iloc[0][Batter + 'RBI'] + 1
                summary.at[0, third  + 'Runs']  = summary.iloc[0][third  + 'Runs'] + 1
                third = second; second = first; first = Batter
            if Pitcher == AwayPitchers[0] or Pitcher == HomePitchers[0]:
                summary.at[0, Pitcher + 'Walks'] = summary.iloc[0][Pitcher + 'Walks'] + 1

        elif result == 'HR':
            # Home run: score batter + all base runners
            runs = 1 + (1 if first != 0 else 0) + (1 if second != 0 else 0) + (1 if third != 0 else 0)
            score = score + runs
            summary.at[0, std    + 'Hits']  = summary.at[0, std    + 'Hits'] + 1
            summary.at[0, std    + 'HR']    = summary.at[0, std    + 'HR']   + 1
            summary.at[0, Batter + 'TB']    = summary.iloc[0][Batter + 'TB'] + 4
            summary.at[0, Batter + 'HRR']   = summary.iloc[0][Batter + 'HRR'] + 1
            summary.at[0, Batter + 'Hits']  = summary.iloc[0][Batter + 'Hits'] + 1
            summary.at[0, Batter + 'RBI']   = summary.iloc[0][Batter + 'RBI'] + runs
            summary.at[0, Batter + 'Runs']  = summary.iloc[0][Batter + 'Runs'] + 1
            if first  != 0: summary.at[0, first  + 'Runs'] = summary.iloc[0][first  + 'Runs'] + 1
            if second != 0: summary.at[0, second + 'Runs'] = summary.iloc[0][second + 'Runs'] + 1
            if third  != 0: summary.at[0, third  + 'Runs'] = summary.iloc[0][third  + 'Runs'] + 1
            first = 0; second = 0; third = 0

        elif result == 'out':
            err = rand.random()
            if err < 0.0468:
                # Error: batter reaches base safely despite out probability
                if first == 0:
                    first = Batter
                elif first != 0 and second == 0:
                    second = first; first = Batter
                elif first != 0 and second != 0 and third == 0:
                    third = second; second = first; first = Batter
                else:
                    score = score + 1
                    third = second; second = first; first = Batter
            else:
                if third != 0 and outs < 2:
                    # Sacrifice fly opportunity with runner on third
                    sf = rand.random()
                    if sf < 0.55:
                        score = score + 1
                        summary.at[0, Batter + 'RBI']  = summary.iloc[0][Batter + 'RBI'] + 1
                        summary.at[0, third  + 'Runs'] = summary.iloc[0][third  + 'Runs'] + 1
                        outs = outs + 1; third = 0
                    else:
                        outs = outs + 1
                elif first != 0:
                    if trj == 'GB':
                        # Ground ball with runner on first: check for double play
                        gidp = rand.random()
                        if gidp < 0.055:
                            outs = outs + 2; first = 0
                        else:
                            outs = outs + 1; first = batter
                    else:
                        outs = outs + 1
                else:
                    outs = outs + 1

        elif result == 'Sin':
            # Single: advance runners based on baserunning rolls
            summary.at[0, std    + 'Hits'] = summary.at[0, std    + 'Hits'] + 1
            summary.at[0, Batter + 'TB']   = summary.iloc[0][Batter + 'TB'] + 1
            summary.at[0, Batter + 'Hits'] = summary.iloc[0][Batter + 'Hits'] + 1
            # (Baserunner advancement logic per base state follows)
            if first == 0 and second == 0 and third == 0:
                first = Batter
            elif first != 0 and second == 0 and third == 0:
                third = first if baserun1 <= 0.31731586 else None
                if third is None: second = first
                first = Batter
            elif first != 0 and second != 0 and third == 0:
                if baserun2 <= 0.61667691:
                    summary.at[0, Batter  + 'RBI']  = summary.iloc[0][Batter  + 'RBI']  + 1
                    summary.at[0, second  + 'Runs'] = summary.iloc[0][second  + 'Runs'] + 1
                    score = score + 1
                    third = first if baserun1 <= 0.31731586 else None
                    if third is None: second = first
                    first = Batter
                else:
                    third = second; second = first; first = Batter
            elif first != 0 and second != 0 and third != 0:
                if baserun2 <= 0.61667691:
                    summary.at[0, Batter  + 'RBI']  = summary.iloc[0][Batter  + 'RBI']  + 2
                    summary.at[0, third   + 'Runs'] = summary.iloc[0][third   + 'Runs'] + 1
                    summary.at[0, second  + 'Runs'] = summary.iloc[0][second  + 'Runs'] + 1
                    score = score + 2; third = 0
                    third = first if baserun1 <= 0.31731586 else None
                    if third is None: second = first
                    first = Batter
                else:
                    summary.at[0, Batter  + 'RBI']  = summary.iloc[0][Batter  + 'RBI']  + 1
                    summary.at[0, third   + 'Runs'] = summary.iloc[0][third   + 'Runs'] + 1
                    score = score + 1
                    third = second; second = first; first = Batter
            elif first == 0 and second != 0 and third != 0:
                if baserun2 <= 0.61667691:
                    summary.at[0, Batter  + 'RBI']  = summary.iloc[0][Batter  + 'RBI']  + 2
                    summary.at[0, third   + 'Runs'] = summary.iloc[0][third   + 'Runs'] + 1
                    summary.at[0, second  + 'Runs'] = summary.iloc[0][second  + 'Runs'] + 1
                    score = score + 2; third = 0; second = 0; first = Batter
                else:
                    summary.at[0, Batter  + 'RBI']  = summary.iloc[0][Batter  + 'RBI']  + 1
                    summary.at[0, third   + 'Runs'] = summary.iloc[0][third   + 'Runs'] + 1
                    score = score + 1; third = second; second = 0; first = Batter
            elif first != 0 and second == 0 and third != 0:
                summary.at[0, Batter + 'RBI']  = summary.iloc[0][Batter + 'RBI']  + 1
                summary.at[0, third  + 'Runs'] = summary.iloc[0][third  + 'Runs'] + 1
                score = score + 1
                if baserun1 <= 0.31731586:
                    third = first; second = 0; first = Batter
                else:
                    third = 0; second = first; first = Batter
            elif first == 0 and second == 0 and third != 0:
                summary.at[0, Batter + 'RBI']  = summary.iloc[0][Batter + 'RBI']  + 1
                summary.at[0, third  + 'Runs'] = summary.iloc[0][third  + 'Runs'] + 1
                score = score + 1; third = 0; first = Batter
            elif first == 0 and second != 0 and third == 0:
                if baserun2 <= 0.61667691:
                    summary.at[0, Batter  + 'RBI']  = summary.iloc[0][Batter  + 'RBI']  + 1
                    summary.at[0, second  + 'Runs'] = summary.iloc[0][second  + 'Runs'] + 1
                    score = score + 1; first = Batter; second = 0
                else:
                    third = second; second = 0; first = batter

        elif result == 'db':
            # Double: lead runner generally scores; trailing runner may hold at third
            summary.at[0, std    + 'Hits'] = summary.at[0, std    + 'Hits'] + 1
            summary.at[0, Batter + 'TB']   = summary.iloc[0][Batter + 'TB'] + 2
            summary.at[0, Batter + 'Hits'] = summary.iloc[0][Batter + 'Hits'] + 1
            if first == 0 and second == 0 and third == 0:
                second = Batter
            elif first != 0 and second == 0 and third == 0:
                if baserun1 <= 0.381262729:
                    summary.at[0, Batter + 'RBI']  = summary.iloc[0][Batter + 'RBI']  + 1
                    summary.at[0, first  + 'Runs'] = summary.iloc[0][first  + 'Runs'] + 1
                    second = Batter; first = 0; score = score + 1
                else:
                    third = first; second = Batter; first = 0
            elif first != 0 and second != 0 and third == 0:
                if baserun1 <= 0.381262729:
                    summary.at[0, Batter  + 'RBI']  = summary.iloc[0][Batter  + 'RBI']  + 2
                    summary.at[0, first   + 'Runs'] = summary.iloc[0][first   + 'Runs'] + 1
                    summary.at[0, second  + 'Runs'] = summary.iloc[0][second  + 'Runs'] + 1
                    score = score + 2; third = 0; first = 0; second = Batter
                else:
                    summary.at[0, Batter  + 'RBI']  = summary.iloc[0][Batter  + 'RBI']  + 1
                    summary.at[0, second  + 'Runs'] = summary.iloc[0][second  + 'Runs'] + 1
                    score = score + 1; third = first; second = Batter; first = 0
            elif first != 0 and second != 0 and third != 0:
                if baserun1 <= 0.381262729:
                    summary.at[0, Batter  + 'RBI']  = summary.iloc[0][Batter  + 'RBI']  + 3
                    summary.at[0, third   + 'Runs'] = summary.iloc[0][third   + 'Runs'] + 1
                    summary.at[0, second  + 'Runs'] = summary.iloc[0][second  + 'Runs'] + 1
                    summary.at[0, first   + 'Runs'] = summary.iloc[0][first   + 'Runs'] + 1
                    score = score + 3; third = 0; second = Batter; first = 0
                else:
                    summary.at[0, Batter  + 'RBI']  = summary.iloc[0][Batter  + 'RBI']  + 2
                    summary.at[0, third   + 'Runs'] = summary.iloc[0][third   + 'Runs'] + 1
                    summary.at[0, second  + 'Runs'] = summary.iloc[0][second  + 'Runs'] + 1
                    score = score + 2; third = first; second = Batter; first = 0
            elif first == 0 and second != 0 and third != 0:
                summary.at[0, Batter  + 'RBI']  = summary.iloc[0][Batter  + 'RBI']  + 2
                summary.at[0, third   + 'Runs'] = summary.iloc[0][third   + 'Runs'] + 1
                summary.at[0, second  + 'Runs'] = summary.iloc[0][second  + 'Runs'] + 1
                score = score + 2; third = 0; second = Batter
            elif first != 0 and second == 0 and third != 0:
                if baserun1 <= 0.381262729:
                    summary.at[0, Batter  + 'RBI']  = summary.iloc[0][Batter  + 'RBI']  + 2
                    summary.at[0, third   + 'Runs'] = summary.iloc[0][third   + 'Runs'] + 1
                    summary.at[0, first   + 'Runs'] = summary.iloc[0][first   + 'Runs'] + 1
                    score = score + 2; third = 0; second = Batter; first = 0
                else:
                    summary.at[0, Batter  + 'RBI']  = summary.iloc[0][Batter  + 'RBI']  + 1
                    summary.at[0, third   + 'Runs'] = summary.iloc[0][third   + 'Runs'] + 1
                    score = score + 1; third = first; second = Batter; first = 0
            elif first == 0 and second == 0 and third != 0:
                summary.at[0, Batter + 'RBI']  = summary.iloc[0][Batter + 'RBI']  + 1
                summary.at[0, third  + 'Runs'] = summary.iloc[0][third  + 'Runs'] + 1
                score = score + 1; third = 0; second = Batter
            elif first == 0 and second != 0 and third == 0:
                summary.at[0, Batter  + 'RBI']  = summary.iloc[0][Batter  + 'RBI']  + 1
                summary.at[0, second  + 'Runs'] = summary.iloc[0][second  + 'Runs'] + 1
                score = score + 1; second = Batter

        elif result == 'trp':
            # Triple: batter to third, all runners score
            summary.at[0, std    + 'Hits'] = summary.at[0, std    + 'Hits'] + 1
            summary.at[0, Batter + 'Hits'] = summary.iloc[0][Batter + 'Hits'] + 1
            summary.at[0, Batter + 'TB']   = summary.iloc[0][Batter + 'TB']   + 3
            if first != 0:
                score = score + 1
                summary.at[0, first  + 'Runs'] = summary.iloc[0][first  + 'Runs'] + 1
                summary.at[0, Batter + 'RBI']  = summary.iloc[0][Batter + 'RBI']  + 1
                first = 0
            if second != 0:
                score = score + 1
                summary.at[0, second + 'Runs'] = summary.iloc[0][second + 'Runs'] + 1
                summary.at[0, Batter + 'RBI']  = summary.iloc[0][Batter + 'RBI']  + 1
                second = 0
            if third != 0:
                score = score + 1
                summary.at[0, third  + 'Runs'] = summary.iloc[0][third  + 'Runs'] + 1
                summary.at[0, Batter + 'RBI']  = summary.iloc[0][Batter + 'RBI']  + 1
            third = Batter

        else:
            print('error')

        # Advance batter and pitcher counters; wrap lineup at 9
        cp = cp + 1
        cb = cb + 1
        if cb > 8:
            cb = 0
        counter = counter + 1

    return score, cp, cb


# ===========================================================================
# Main simulation loop: iterate over all scheduled games
# ===========================================================================
games = pd.read_csv('MLBGames.csv')
games.index = games['Away']

for x in games['Away'][10:]:

    # -----------------------------------------------------------------------
    # Set up game-level variables
    # -----------------------------------------------------------------------
    Away = games.loc[x]['Away']
    Home = games.loc[x]['Home']
    sheetname = Away + Home
    shortname = sheetname[:30] + '.csv'
    print(shortname)

    # Pull lineups for this matchup
    Awaybatters  = Battinglineups[Away]
    HomeBatters  = Battinglineups[Home]
    AwayPitchers = Pitchlineups[Away]
    HomePitchers = Pitchlineups[Home]

    # Temperature adjustment: degrees above/below seasonal average at home park
    # Each coefficient was derived empirically from historical rate differences by temperature
    Exceessdegrees = Weather.loc[Home]['Temp'] - Weather.loc[Home]['AverageTemp']
    TempAdjSo   = Exceessdegrees * -0.000432758  # higher temp = fewer strikeouts
    TempAdjSin  = Exceessdegrees *  0.000245289  # higher temp = more singles
    TempAdjDb   = Exceessdegrees *  0.000125241  # higher temp = more doubles
    TempAdjHR   = Exceessdegrees *  0.000165814  # higher temp = more home runs
    TempADjHard = Exceessdegrees *  0.000414407  # higher temp = more hard contact
    TempAdjSoft = Exceessdegrees * -0.000580972  # higher temp = less soft contact

    # Wind adjustment: directional effect on HR rate
    inout = Weather.loc[Home]['Wind']     # 'In', 'Out', or crosswind
    spd   = Weather.loc[Home]['WindMag']  # wind speed magnitude

    if inout == 'Out':
        WindAdHR = spd *  0.000269773   # wind out boosts HR
    elif inout == 'In':
        WindAdHR = spd * -0.000269773   # wind in suppresses HR
    else:
        WindAdHR = 0

    # -----------------------------------------------------------------------
    # Run 1,000 Monte Carlo simulations for this matchup
    # -----------------------------------------------------------------------
    game = 1
    while game < 1001:

        # Reset inning and pitch/batter counters for each simulation
        inning = 1
        AwayCP = 0   # away pitcher cumulative batters faced
        HomeCP = 0   # home pitcher cumulative batters faced
        AwayCB = 0   # away batting lineup position
        HomeCB = 0   # home batting lineup position

        # Starter expected batters faced before relief (BF stat)
        AwayPC = pitchersdata.loc[AwayPitchers[0]]['BF']
        HomePC = pitchersdata.loc[HomePitchers[0]]['BF']

        # Build output column list for this game's summary DataFrame
        sumcols = []
        counter = 1
        while counter < 16:
            sumcols.append(Away + str(counter)); counter = counter + 1
        counter = 1
        while counter < 16:
            sumcols.append(Home + str(counter)); counter = counter + 1

        # Team-level totals
        sumcols += [Away + 'Hits', Home + 'Hits', Away + 'HR', Home + 'HR',
                    Away + 'SO',   Home + 'SO',
                    AwayPitchers[0] + 'SO',    HomePitchers[0] + 'SO',
                    AwayPitchers[0] + 'Walks', HomePitchers[0] + 'Walks']

        # Per-batter stat columns
        for batter in list(Awaybatters) + list(HomeBatters):
            for stat in ['TB', 'Runs', 'RBI', 'Hits', 'HRR']:
                sumcols.append(batter + stat)

        awaytotal = 0
        hometotal = 0
        summary = pd.DataFrame(0, index=np.arange(0, 1), columns=sumcols)
        summary = summary.loc[:, ~summary.columns.duplicated()]

        # -----------------------------------------------------------------------
        # Simulate innings 1-9 (plus extras if tied)
        # -----------------------------------------------------------------------
        while inning < 10:
            # Away half-inning
            score   = runoffense('away', HomeCP, HomePC, AwayCB)
            result  = score[0]; HomeCP = score[1]; AwayCB = score[2]
            summary.at[0, Away + str(inning)] = result
            awaytotal = awaytotal + result

            # Home half-inning (skip bottom of 9th if away team is losing)
            if inning < 9:
                score    = runoffense('home', AwayCP, AwayPC, HomeCB)
                result   = score[0]; AwayCP = score[1]; HomeCB = score[2]
                hometotal = hometotal + result
            else:
                if awaytotal < hometotal:
                    result = 0   # home team wins; no need to bat
                else:
                    score    = runoffense('home', AwayCP, AwayPC, HomeCB)
                    result   = score[0]; AwayCP = score[1]; HomeCB = score[2]

            summary.at[0, Home + str(inning)] = result
            inning = inning + 1

            # Extra innings if tied after 9 (up to 12)
            if awaytotal == hometotal:
                inning = inning + 1
                while awaytotal == hometotal and inning < 13:
                    score     = runoffense('away', HomeCP, HomePC, AwayCB)
                    result    = score[0]; HomeCP = score[1]; AwayCB = score[2]
                    summary.at[0, Away + str(inning)] = result
                    awaytotal = awaytotal + result

                    score     = runoffense('home', AwayCP, AwayPC, HomeCB)
                    result    = score[0]; AwayCP = score[1]; HomeCB = score[2]
                    summary.at[0, Home + str(inning)] = result
                    hometotal = hometotal + result
                    inning    = inning + 1

        # Accumulate simulation results
        if game == 1:
            Final = summary
        else:
            Final = Final.append(summary)
        game = game + 1

    # Write 1,000-game simulation output for this matchup to CSV
    Final.to_csv(outdir + shortname, encoding='cp1252')
