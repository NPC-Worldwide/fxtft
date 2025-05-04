# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import time
import os
import sys

# Use standard fonts
plt.rcParams.update({
    "text.usetex": False, "font.family": "serif", "axes.labelsize": 12,
    "font.size": 11, "legend.fontsize": 10, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "figure.titlesize": 14
})

# Simulation parameters
NUM_ROUNDS = 500
NUM_SIMULATIONS_PER_MATCHUP = 100 # Adjust for desired simulation depth
NOISE_LEVEL = 0.05

# --- Move Constants and Payoff Matrix (using integers) ---
COOPERATE = 1
DEFECT = 0


# Payoff matrix: P[my_move, opponent_move] -> (my_score, opponent_score)
# Rows: My move (0=D, 1=C), Columns: Opponent's move (0=D, 1=C)
PAYOFF_MATRIX = np.array([
    [(1, 1), (5, 0)],  # My move D: (D,D) -> (1,1), (D,C) -> (5,0)
    [(0, 5), (3, 3)]   # My move C: (C,D) -> (0,5), (C,C) -> (3,3)
])

# --- Core Strategy Definitions (returning COOPERATE/DEFECT) ---

def always_cooperate(history, opponent_history):
    return COOPERATE

def always_defect(history, opponent_history):
    return DEFECT

def tit_for_tat(history, opponent_history):
    if not opponent_history: # Check if list is empty
        return COOPERATE
    # Opponent history is list of integers (0 or 1)
    return opponent_history[-1]

def tit_for_two_tats(history, opponent_history):
    if len(opponent_history) < 2:
        return COOPERATE
    # Check last two moves for defection (0)
    if opponent_history[-1] == DEFECT and opponent_history[-2] == DEFECT:
        return DEFECT
    return COOPERATE

def random_strategy(history, opponent_history):
    return COOPERATE if np.random.random() < 0.5 else DEFECT

# --- Renamed Strategy: The Forgiving Retaliator (FR) ---

class ForgivingRetaliator: # RENAMED CLASS
    def __init__(self, lookback=10):
        self.lookback = lookback
        self.punishment_turns_remaining = 0
        self.next_punishment_duration = 1

    def __call__(self, history, opponent_history):
        current_round = len(history)

        if self.punishment_turns_remaining > 0:
            self.punishment_turns_remaining -= 1
            return DEFECT

        if current_round < self.lookback:
            return COOPERATE
        else:
            # Convert recent history slice to NumPy array for efficient check
            # Ensure opponent_history contains integers
            recent_opponent_history = np.array(opponent_history[-self.lookback:], dtype=int)
            num_defections = np.sum(recent_opponent_history == DEFECT)

            if num_defections >= 2:
                punishment_duration_this_time = self.next_punishment_duration
                self.punishment_turns_remaining = punishment_duration_this_time
                self.next_punishment_duration *= 2

                if self.punishment_turns_remaining > 0:
                    self.punishment_turns_remaining -= 1
                    return DEFECT
                else:
                    return COOPERATE # Should not happen if duration starts >= 1
            else:
                self.next_punishment_duration = 1
                return COOPERATE

    def __repr__(self):
        # UPDATED representation to reflect new name (can be simpler)
        return f"ForgivingRetaliator(L={self.lookback})"

# --- Simulation Core (Sequential, NumPy where applicable) ---


# --- New Strategy: Severity Punisher (SP) ---
class SeverityPunisher:
    def __init__(self, lookback=10, threshold=2, base=4):
        self.lookback = lookback
        self.threshold = threshold # Min defects to trigger *any* punishment
        self.base = base # Base for the geometric series (e.g., 4)
        self.punishment_turns_remaining = 0
        # No next_punishment_duration state needed

    def __call__(self, history, opponent_history):
        current_round = len(history)

        # Serve any ongoing punishment
        if self.punishment_turns_remaining > 0:
            self.punishment_turns_remaining -= 1
            return DEFECT

        # Cooperate initially
        if current_round < self.lookback:
            return COOPERATE

        # Analyze recent history
        recent_opponent_history = np.array(opponent_history[-self.lookback:], dtype=int)
        num_defections = np.sum(recent_opponent_history == DEFECT)

        # Check threshold
        if num_defections >= self.threshold:
            # Calculate punishment duration based on severity (number of defects)
            # Number of terms in the sum: based on pairs of defections
            num_terms = num_defections // 2
            if num_terms > 0:
                 # Calculate sum of geometric series: base^0 + base^1 + ... + base^(num_terms-1)
                 # Efficient calculation using formula if base != 1: (base^num_terms - 1) // (base - 1)
                 # Or direct sum for clarity / robustness if base could be 1
                 # duration = np.sum(self.base ** np.arange(num_terms)) # Direct sum
                 if self.base == 1:
                     duration = num_terms # Sum of 1s
                 else:
                     duration = (self.base**num_terms - 1) // (self.base - 1) # Formula
            else:
                # This handles the case where threshold is >=2 but num_defects might be odd (e.g., 3)
                # If threshold=2, num_defects=3 -> num_terms=1 -> duration=1
                # If threshold=2, num_defects=2 -> num_terms=1 -> duration=1
                # Set minimum duration if needed, but formula handles base cases.
                # If num_terms is 0 (e.g. threshold=1, num_defects=1), duration would be 0.
                # Let's ensure at least 1 round if threshold met, matching user example for d=2,3 -> 1
                if num_defects >= self.threshold:
                    num_terms = max(1, num_defects // 2) # Ensure at least 1 term if threshold met
                    if self.base == 1: duration = num_terms
                    else: duration = (self.base**num_terms - 1) // (self.base - 1)
                else:
                    duration = 0 # Should not happen if threshold check is correct

            # Cap duration? 341 rounds is longer than many games.
            max_punishment = NUM_ROUNDS # Cap at game length
            duration = min(duration, max_punishment)

            self.punishment_turns_remaining = duration

            # Start punishment immediately if duration > 0
            if self.punishment_turns_remaining > 0:
                self.punishment_turns_remaining -= 1
                return DEFECT
            else:
                return COOPERATE # If calculated duration is 0
        else:
            # Opponent behaved well enough, cooperate
            return COOPERATE

    def __repr__(self):
         # Include base in representation
        return f"SeverityPunisher(L={self.lookback},T={self.threshold},B={self.base})"


# --- Simulation Core & Analysis Functions ---
# (run_game, run_tournament, analyze_dynamics, analyze_noise_impact - assumed unchanged)
def run_game(strategy1_func, strategy2_func, rounds=NUM_ROUNDS, add_noise=False, noise_level=NOISE_LEVEL):
    """Runs a single game sequentially."""
    s1 = strategy1_func() if isinstance(strategy1_func, type) else strategy1_func
    s2 = strategy2_func() if isinstance(strategy2_func, type) else strategy2_func
    history1, history2 = [], []
    scores = np.zeros(2, dtype=int)
    for _ in range(rounds):
        move1 = s1(history1.copy(), history2.copy()); move2 = s2(history2.copy(), history1.copy())
        if add_noise:
            if np.random.random() < noise_level: move1 = 1 - move1
            if np.random.random() < noise_level: move2 = 1 - move2
        history1.append(move1); history2.append(move2)
        payoff1, payoff2 = PAYOFF_MATRIX[move1, move2]
        scores[0] += payoff1; scores[1] += payoff2
    hist1_arr = np.array(history1, dtype=int); hist2_arr = np.array(history2, dtype=int)
    coop_rate1 = np.mean(hist1_arr) if rounds > 0 else 0; coop_rate2 = np.mean(hist2_arr) if rounds > 0 else 0
    return {'score1': scores[0], 'score2': scores[1],'cooperation_rate1': coop_rate1, 'cooperation_rate2': coop_rate2, 'history1': history1, 'history2': history2}

def run_tournament(strategies_dict, trials_per_matchup=NUM_SIMULATIONS_PER_MATCHUP, rounds_per_game=NUM_ROUNDS, add_noise=False, noise_level=NOISE_LEVEL):
    """Runs an all-play-all tournament sequentially."""
    strategy_names = list(strategies_dict.keys()); num_strategies = len(strategy_names)
    results = {name: {'total_score': 0, 'games': 0, 'avg_score': 0,'wins': 0, 'losses': 0, 'ties': 0, 'cooperation_rate': 0.0} for name in strategy_names}
    total_matchups = num_strategies * num_strategies; total_trials_overall = total_matchups * trials_per_matchup
    desc = f"Noise={noise_level:.2f}" if add_noise else "No Noise"
    print(f"Running tournament ({desc}, {total_trials_overall:,} total games)...")
    start_compute_time = time.time(); matchup_count = 0
    for i in range(num_strategies):
        for j in range(num_strategies):
            name1, name2 = strategy_names[i], strategy_names[j]; strat1_def, strat2_def = strategies_dict[name1], strategies_dict[name2]
            matchup_count += 1
            if matchup_count % max(1, total_matchups // 20) == 0 or matchup_count == 1:
                 elapsed_time = time.time() - start_compute_time; print(f"  Matchup {matchup_count}/{total_matchups} ({name1} vs {name2})... [{elapsed_time:.1f}s elapsed]", end='\r'); sys.stdout.flush()
            for _ in range(trials_per_matchup):
                game_result = run_game(strat1_def, strat2_def, rounds=rounds_per_game, add_noise=add_noise, noise_level=noise_level)
                score1, score2 = game_result['score1'], game_result['score2']; coop1, coop2 = game_result['cooperation_rate1'], game_result['cooperation_rate2']
                results[name1]['total_score'] += score1; results[name2]['total_score'] += score2
                results[name1]['games'] += 1; results[name2]['games'] += 1
                results[name1]['cooperation_rate'] += coop1; results[name2]['cooperation_rate'] += coop2
                if score1 > score2: results[name1]['wins'] += 1; results[name2]['losses'] += 1
                elif score1 < score2: results[name1]['losses'] += 1; results[name2]['wins'] += 1
                else: results[name1]['ties'] += 1; results[name2]['ties'] += 1
    print(); end_compute_time = time.time(); print(f"Tournament finished in {end_compute_time - start_compute_time:.2f} seconds.")
    print("Calculating final tournament averages...");
    for name in strategy_names:
        games_played = results[name]['games']
        if games_played > 0:
            results[name]['avg_score'] = results[name]['total_score'] / games_played
            results[name]['cooperation_rate'] = results[name]['cooperation_rate'] / games_played
            total_outcomes = results[name]['wins'] + results[name]['losses'] + results[name]['ties']
            results[name]['win_rate'] = results[name]['wins'] / total_outcomes if total_outcomes > 0 else 0
        else: results[name]['avg_score'] = 0; results[name]['cooperation_rate'] = 0; results[name]['win_rate'] = 0
    return results

def analyze_dynamics(strategy1_def, strategy2_def, strategy1_name, strategy2_name, rounds=NUM_ROUNDS, noise=False, noise_level=NOISE_LEVEL):
    s1 = strategy1_def() if isinstance(strategy1_def, type) else strategy1_def; s2 = strategy2_def() if isinstance(strategy2_def, type) else strategy2_def
    result = run_game(s1, s2, rounds=rounds, add_noise=noise, noise_level=noise_level)
    history1_arr = np.array(result['history1'], dtype=int); history2_arr = np.array(result['history2'], dtype=int)
    round_payoffs1 = np.zeros(rounds, dtype=int); round_payoffs2 = np.zeros(rounds, dtype=int)
    for r in range(rounds): p1, p2 = PAYOFF_MATRIX[history1_arr[r], history2_arr[r]]; round_payoffs1[r], round_payoffs2[r] = p1, p2
    running_score1 = np.cumsum(round_payoffs1); running_score2 = np.cumsum(round_payoffs2)
    coop_counts1, coop_counts2 = np.cumsum(history1_arr), np.cumsum(history2_arr)
    round_numbers = np.arange(1, rounds + 1); running_coop1, running_coop2 = coop_counts1 / round_numbers, coop_counts2 / round_numbers
    return {'running_score1': running_score1.tolist(), 'running_score2': running_score2.tolist(), 'running_coop1': running_coop1.tolist(), 'running_coop2': running_coop2.tolist(), 'final_score1': running_score1[-1] if rounds > 0 else 0, 'final_score2': running_score2[-1] if rounds > 0 else 0, 'history1': result['history1'], 'history2': result['history2']}

def analyze_noise_impact(strategies_dict, noise_levels, trials_per_matchup=NUM_SIMULATIONS_PER_MATCHUP // 10, rounds_per_game=NUM_ROUNDS):
    strategy_names = list(strategies_dict.keys()); results = {name: {'avg_scores': [], 'coop_rates': []} for name in strategy_names}
    actual_trials = max(10, trials_per_matchup); print(f"\nAnalyzing noise impact sequentially (using {actual_trials} trials per matchup)...")
    for noise in noise_levels:
        print(f"  Noise level: {noise:.2f}"); sys.stdout.flush(); add_noise = noise > 0
        tournament_results = run_tournament(strategies_dict, trials_per_matchup=actual_trials, rounds_per_game=rounds_per_game, add_noise=add_noise, noise_level=noise)
        for name in strategy_names:
            if name in tournament_results: results[name]['avg_scores'].append(tournament_results[name].get('avg_score', 0)); results[name]['coop_rates'].append(tournament_results[name].get('cooperation_rate', 0))
            else: results[name]['avg_scores'].append(0); results[name]['coop_rates'].append(0); print(f"Warning: Strategy {name} missing results at noise {noise:.2f}")
    return results, noise_levels



# --- Plotting Functions (Integrated and Adapted for Integer Moves) ---
# Note: Labels within plots will now use the dictionary key name ('ForgivingRetaliator')

def plot_tournament_results(results, save_path=None):
    """ Plots tournament results bar chart (scores and cooperation rate). """
    strategy_names = list(results.keys())
    if not strategy_names:
        print("Warning: No results to plot in plot_tournament_results.")
        return

    # Average score per game (total score / total games)
    avg_scores_per_game = [results[name].get('avg_score', 0) for name in strategy_names]
    # Average score per round (avg score per game / num rounds)
    avg_scores_per_round = [score / NUM_ROUNDS for score in avg_scores_per_game]

    coop_rates = [results[name].get('cooperation_rate', 0) for name in strategy_names]

    # Win rates
    win_rates = []
    for name in strategy_names:
        wins = results[name].get('wins', 0)
        losses = results[name].get('losses', 0)
        ties = results[name].get('ties', 0)
        total_outcomes = wins + losses + ties
        win_rate = wins / total_outcomes if total_outcomes > 0 else 0
        win_rates.append(win_rate)

    # Sort by average score per round
    try:
        if not all(isinstance(s, (int, float)) for s in avg_scores_per_round):
             print("Warning: Non-numeric scores found, skipping sorting.")
        else:
             sorted_indices = np.argsort(avg_scores_per_round)[::-1]
             strategy_names = [strategy_names[i] for i in sorted_indices]
             avg_scores_per_round = [avg_scores_per_round[i] for i in sorted_indices]
             coop_rates = [coop_rates[i] for i in sorted_indices]
             win_rates = [win_rates[i] for i in sorted_indices]
    except Exception as e:
        print(f"Error sorting results for plotting: {e}")
        pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Prisoner\'s Dilemma Strategy Comparison', fontsize=16, fontweight='bold')
    bar_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(strategy_names))) # Use a colormap
    win_rate_color = '#e74c3c'

    # --- Average Score Per Round Plot ---
    ax1.bar(strategy_names, avg_scores_per_round, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Average Score per Round', fontsize=12, fontweight='bold')
    ax1.set_title('Strategy Performance', fontsize=14, fontweight='bold')
    max_score_val = max(avg_scores_per_round) if avg_scores_per_round else 1
    # Set ylim based on possible payoff range (0 to 5)
    ax1.set_ylim(0, max(PAYOFF_MATRIX.max(), max_score_val * 1.1))
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for i, score in enumerate(avg_scores_per_round):
        offset = 0.03 * max(1, max_score_val)
        ax1.text(i, score + offset, f'{score:.2f}', ha='center', va='bottom', fontweight='bold', color='black', fontsize=9)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=10, fontweight='bold')

    # --- Cooperation Rate Plot ---
    ax2.bar(strategy_names, coop_rates, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Cooperation Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Strategy Cooperation', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for i, rate in enumerate(coop_rates):
        ax2.text(i, rate + 0.05, f'{rate:.2f}', ha='center', va='bottom', fontweight='bold', color='black', fontsize=9)

    # --- Win Rate Plot (on second axis) ---
    ax3 = ax2.twinx()
    ax3.plot(strategy_names, win_rates, 'o-', color=win_rate_color, linewidth=3, markersize=8, markeredgecolor='black', markeredgewidth=1, label='Win Rate')
    for i, rate in enumerate(win_rates):
        ax3.text(i, rate + 0.03, f'{rate:.2f}', ha='center', va='bottom', fontweight='bold', color=win_rate_color, fontsize=9)
    ax3.set_ylabel('Win Rate', fontsize=12, fontweight='bold', color=win_rate_color)
    ax3.tick_params(axis='y', labelcolor=win_rate_color, labelsize=10)
    ax3.set_ylim(0, 1.1)
    ax3.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    if save_path:
        try: plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"Plot saved to {save_path}")
        except Exception as e: print(f"Error saving plot {save_path}: {e}")
    else: plt.show()
    plt.close(fig)

def plot_dynamics(dynamics, strategy1_name, strategy2_name, save_path=None):
    """ Plots dynamics between two strategies (scores and cooperation). """
    if not dynamics or 'running_score1' not in dynamics or not dynamics['running_score1']:
         print(f"Warning: Invalid dynamics data for {strategy1_name} vs {strategy2_name}.")
         return
    rounds = len(dynamics['running_score1'])
    if rounds == 0: return
    rounds_axis = np.arange(1, rounds + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True) # Slightly smaller height
    # Use the names passed to the function for the title
    fig.suptitle(f'Strategy Dynamics: {strategy1_name} vs {strategy2_name}', fontsize=16, fontweight='bold')
    color1, color2 = '#3498db', '#e74c3c'
    marker_interval = max(rounds // 10, 1)

    # Cumulative scores
    running_score1 = np.array(dynamics['running_score1'])
    running_score2 = np.array(dynamics['running_score2'])
    # Use the names passed to the function for labels
    ax1.plot(rounds_axis, running_score1, color=color1, linewidth=2.5, label=strategy1_name)
    ax1.plot(rounds_axis, running_score2, color=color2, linewidth=2.5, label=strategy2_name)
    ax1.plot(rounds_axis[::marker_interval], running_score1[::marker_interval], 'o', color=color1, ms=7, mec='black', mew=0.5)
    ax1.plot(rounds_axis[::marker_interval], running_score2[::marker_interval], 's', color=color2, ms=7, mec='black', mew=0.5)
    ax1.set_ylabel('Cumulative Score', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=11, frameon=True, facecolor='white', edgecolor='black')
    # Final score annotations (simpler)
    final_score1 = running_score1[-1]
    final_score2 = running_score2[-1]
    ax1.text(rounds, final_score1, f' {final_score1}', color=color1, ha='left', va='center', fontsize=10, fontweight='bold')
    ax1.text(rounds, final_score2, f' {final_score2}', color=color2, ha='left', va='center', fontsize=10, fontweight='bold')

    # Cooperation rates
    running_coop1 = np.array(dynamics['running_coop1'])
    running_coop2 = np.array(dynamics['running_coop2'])
    # Use the names passed to the function for labels
    ax2.plot(rounds_axis, running_coop1, color=color1, linewidth=2.5, label=f'{strategy1_name} Coop Rate')
    ax2.plot(rounds_axis, running_coop2, color=color2, linewidth=2.5, label=f'{strategy2_name} Coop Rate')
    ax2.plot(rounds_axis[::marker_interval], running_coop1[::marker_interval], 'o', color=color1, ms=7, mec='black', mew=0.5)
    ax2.plot(rounds_axis[::marker_interval], running_coop2[::marker_interval], 's', color=color2, ms=7, mec='black', mew=0.5)
    ax2.set_xlabel('Round', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Running Cooperation Rate', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=11, frameon=True, facecolor='white', edgecolor='black')
    ax2.set_ylim(-0.05, 1.05)
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
    # Final coop rate annotations (simpler)
    final_coop1 = running_coop1[-1]
    final_coop2 = running_coop2[-1]
    ax2.text(rounds, final_coop1, f' {final_coop1:.2f}', color=color1, ha='left', va='center', fontsize=10, fontweight='bold')
    ax2.text(rounds, final_coop2, f' {final_coop2:.2f}', color=color2, ha='left', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.subplots_adjust(top=0.93, hspace=0.15)
    if save_path:
        try: plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"Plot saved to {save_path}")
        except Exception as e: print(f"Error saving plot {save_path}: {e}")
    else: plt.show()
    plt.close(fig)

def plot_cooperation_heatmap(history1_int, history2_int, strategy1_name, strategy2_name, save_path=None):
    """ Plot cooperation/defection patterns (expects integer history 0/1). """
    if not isinstance(history1_int, list) or not isinstance(history2_int, list) or not history1_int or not history2_int:
        print(f"Warning: Invalid history for heatmap {strategy1_name} vs {strategy2_name}.")
        return
    rounds = len(history1_int)
    if rounds == 0: return

    # History is already 0s and 1s. 1=C (Green), 0=D (Red)
    data1 = np.array(history1_int)
    data2 = np.array(history2_int)
    data = np.array([data1, data2]) # Shape (2, rounds)

    plt.figure(figsize=(14, 4))
    # RdYlGn: Red (low=0=D) -> Yellow -> Green (high=1=C)
    cmap = plt.cm.get_cmap('RdYlGn', 2)
    im = plt.imshow(data, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=1,
                  extent=[-0.5, rounds - 0.5, 1.5, -0.5]) # Extent for pixel centers

    # Colorbar: Place ticks in the middle of the color segments
    cbar = plt.colorbar(im, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Defect (0)', 'Cooperate (1)'], fontsize=10, fontweight='bold')
    cbar.set_label('Strategy Move', fontsize=12, fontweight='bold')

    # Use the names passed to the function for y-labels
    plt.yticks([0, 1], [strategy1_name, strategy2_name], fontsize=12, fontweight='bold')
    # Use the names passed to the function for the title
    plt.title(f'Cooperation (Green=1) vs Defection (Red=0): {strategy1_name} vs {strategy2_name}',
            fontsize=14, fontweight='bold')
    plt.xlabel('Round', fontsize=14, fontweight='bold')

    # X-axis ticks and grid
    tick_step = max(1, rounds // 20) # Adjust density
    plt.xticks(np.arange(0, rounds, step=tick_step), fontsize=8)
    plt.grid(axis='x', color='white', linestyle='-', linewidth=0.5, alpha=0.3)

    # Statistics Text Box - use names passed to function
    mutual_coop = np.mean((data1 == COOPERATE) & (data2 == COOPERATE))
    mutual_defect = np.mean((data1 == DEFECT) & (data2 == DEFECT))
    coop_rate1 = np.mean(data1)
    coop_rate2 = np.mean(data2)
    stats_text = (f"{strategy1_name}: Coop={coop_rate1:.2f}\n"
                  f"{strategy2_name}: Coop={coop_rate2:.2f}\n"
                  f"Mutual C: {mutual_coop:.2f}\n"
                  f"Mutual D: {mutual_defect:.2f}")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='black')
    # Place text outside the main plot area
    plt.text(1.02, 0.5, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='center', bbox=props)

    # Adjust layout to prevent text box overlap
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # rect=[left, bottom, right, top]

    if save_path:
        try: plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"Plot saved to {save_path}")
        except Exception as e: print(f"Error saving plot {save_path}: {e}")
    else: plt.show()
    plt.close()

def plot_noise_impact(noise_results, noise_levels, save_path=None):
    """ Plot noise impact on strategy scores and cooperation rates. """
    strategy_names = list(noise_results.keys())
    if not strategy_names:
         print("Warning: No noise results to plot.")
         return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Impact of Noise on Strategy Performance', fontsize=16, fontweight='bold')
    # Use a consistent color map based on the original strategies dictionary order if possible
    # For simplicity, using tab10 cycling here
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategy_names)))

    # --- Score vs Noise ---
    max_avg_score_overall = 1.0
    for i, name in enumerate(strategy_names):
        scores = noise_results[name]['avg_scores']
        if not scores: continue
        max_avg_score_overall = max(max_avg_score_overall, max(scores))
        # Plot average score PER GAME
        ax1.plot(noise_levels, scores, 'o-', linewidth=2.5, markersize=7, label=name,
                 color=colors[i], markeredgecolor='black', markeredgewidth=0.5)
    ax1.set_xlabel('Noise Level (Probability of move flip)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Score per Game', fontsize=12, fontweight='bold')
    ax1.set_title('Strategy Performance vs. Noise', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='black', loc='best')
    ax1.set_ylim(bottom=min(0, np.min(PAYOFF_MATRIX)*NUM_ROUNDS*0.9))
    ax1.set_xlim(left=min(noise_levels)-0.01, right=max(noise_levels)+0.01)

    # --- Coop Rate vs Noise ---
    for i, name in enumerate(strategy_names):
        coop_rates = noise_results[name]['coop_rates']
        if not coop_rates: continue
        ax2.plot(noise_levels, coop_rates, 'o-', linewidth=2.5, markersize=7, label=name,
                 color=colors[i], markeredgecolor='black', markeredgewidth=0.5)
    ax2.set_xlabel('Noise Level', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cooperation Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Cooperation Rate vs. Noise', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    # ax2.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='black', loc='best') # Legend might clutter
    ax2.set_ylim(-0.05, 1.05)
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
    ax2.set_xlim(left=min(noise_levels)-0.01, right=max(noise_levels)+0.01)

    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    if save_path:
        try: plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"Plot saved to {save_path}")
        except Exception as e: print(f"Error saving plot {save_path}: {e}")
    else: plt.show()
    plt.close(fig)

def plot_strategy_head_to_head(strategies, trials=30, noise=False, noise_level=NOISE_LEVEL, rounds=NUM_ROUNDS, save_path=None):
    """ Head-to-head matrix (average score per round). Uses fewer trials. """
    strategy_names = list(strategies.keys()) # Get current names from dict
    n_strategies = len(strategy_names)
    print(f"\nGenerating Head-to-Head Matrix ({trials} trials each, sequential)...")
    score_matrix = np.zeros((n_strategies, n_strategies))

    start_h2h_time = time.time()
    for i in range(n_strategies):
        for j in range(n_strategies):
             strat1_def = strategies[strategy_names[i]]
             strat2_def = strategies[strategy_names[j]]
             total_score1 = 0
             for _ in range(trials):
                 game_res = run_game(strat1_def, strat2_def, rounds=rounds, add_noise=noise, noise_level=noise_level)
                 total_score1 += game_res['score1']
             # Store average score PER ROUND for the row player
             score_matrix[i, j] = (total_score1 / trials) / rounds
    print(f"Head-to-Head calculation time: {time.time() - start_h2h_time:.2f}s")


    plt.figure(figsize=(max(8, n_strategies * 1.1), max(6, n_strategies * 0.9)))
    cmap = plt.cm.viridis
    im = plt.imshow(score_matrix, cmap=cmap, interpolation='nearest', aspect='equal')

    vmin, vmax = np.nanmin(score_matrix), np.nanmax(score_matrix)
    midpoint = vmin + (vmax - vmin) / 2 if not np.isnan(vmin) else np.mean(PAYOFF_MATRIX)

    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Average Score per Round (Row vs Column)', fontsize=11, fontweight='bold')

    for i in range(n_strategies):
        for j in range(n_strategies):
            score_val = score_matrix[i, j]
            text_color = 'white' if score_val < midpoint else 'black'
            plt.text(j, i, f'{score_val:.2f}', ha='center', va='center', color=text_color,
                     fontsize=max(6, 9 - n_strategies // 4), fontweight='bold')

    # Use the current strategy names for labels
    plt.xticks(range(n_strategies), strategy_names, rotation=45, ha='right', fontsize=10, fontweight='bold')
    plt.yticks(range(n_strategies), strategy_names, fontsize=10, fontweight='bold')
    plt.xlabel('Column Strategy (Opponent)', fontsize=12, fontweight='bold')
    plt.ylabel('Row Strategy (Player)', fontsize=12, fontweight='bold')
    title = 'Head-to-Head Strategy Comparison (Avg Score/Round)' + (' (with Noise)' if noise else '')
    plt.title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save_path:
        try: plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"Plot saved to {save_path}")
        except Exception as e: print(f"Error saving plot {save_path}: {e}")
    else: plt.show()
    plt.close()

def plot_evolutionary_stability(strategies, rounds=100, population_size=500, generations=500,
                              mutation_rate=0.05, games_per_individual=5, save_path=None):
    """ Evolutionary simulation (simplified for speed). """
    strategy_names = list(strategies.keys()) # Get current names
    n_strategies = len(strategy_names)
    print(f"\nRunning Evolutionary Stability Simulation ({generations} generations, seq)...")
    start_evo_time = time.time()

    def calculate_fitness(individual_idx, current_population):
        my_strategy_idx = current_population[individual_idx]
        my_strat_def = strategies[strategy_names[my_strategy_idx]] # Use current names
        total_score = 0
        opponents_played = 0
        possible_opponents = [idx for idx in range(population_size) if idx != individual_idx]
        if not possible_opponents: return 0
        opponent_indices = np.random.choice(possible_opponents, min(games_per_individual, len(possible_opponents)), replace=False)

        for opp_idx in opponent_indices:
            opponent_strategy_idx = current_population[opp_idx]
            opp_strat_def = strategies[strategy_names[opponent_strategy_idx]] # Use current names
            s1 = my_strat_def() if isinstance(my_strat_def, type) else my_strat_def
            s2 = opp_strat_def() if isinstance(opp_strat_def, type) else opp_strat_def
            result = run_game(s1, s2, rounds=rounds, add_noise=False)
            total_score += result['score1']
            opponents_played += 1
        return total_score / opponents_played if opponents_played > 0 else 0

    population = [i % n_strategies for i in range(population_size)]
    np.random.shuffle(population)
    population_history = np.zeros((generations + 1, n_strategies))

    for gen in range(generations + 1):
        population_history[gen, :] = np.bincount(population, minlength=n_strategies)
        if gen == generations: break
        if gen % max(1, generations // 10) == 0: print(f"  Generation {gen}/{generations}...")

        fitness = np.array([calculate_fitness(i, population) for i in range(population_size)])
        min_fitness = np.min(fitness)
        normalized_fitness = fitness - min_fitness + 1e-6
        total_fitness = np.sum(normalized_fitness)
        probabilities = normalized_fitness / total_fitness if total_fitness > 0 else np.ones(population_size) / population_size
        parent_indices = np.random.choice(range(population_size), size=population_size, replace=True, p=probabilities)
        new_population = []
        for parent_idx in parent_indices:
            if np.random.random() < mutation_rate:
                current_strategy = population[parent_idx]
                possible_mutations = [i for i in range(n_strategies) if i != current_strategy]
                new_strategy = np.random.choice(possible_mutations) if possible_mutations else current_strategy
                new_population.append(new_strategy)
            else:
                new_population.append(population[parent_idx])
        population = new_population
    print(f"Evolutionary sim time: {time.time() - start_evo_time:.2f}s")


    plt.figure(figsize=(12, 7))
    population_pct = population_history / population_size * 100
    colors = plt.cm.tab10(np.linspace(0, 1, n_strategies))
    generations_axis = np.arange(generations + 1)
    for i in range(n_strategies):
        # Use current strategy names for labels
        plt.plot(generations_axis, population_pct[:, i], '-', linewidth=2.5, label=strategy_names[i], color=colors[i])

    plt.xlabel('Generation', fontsize=14, fontweight='bold')
    plt.ylabel('Population Percentage (%)', fontsize=14, fontweight='bold')
    plt.title(f'Evolutionary Dynamics ({population_size} agents, {mutation_rate*100:.1f}% mutation)', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='black', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(-5, 105)
    plt.xlim(-generations*0.02, generations*1.02)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        try: plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"Plot saved to {save_path}")
        except Exception as e: print(f"Error saving plot {save_path}: {e}")
    else: plt.show()
    plt.close()

def plot_recovery_after_defection(strategies, lookback=10, recovery_rounds=30, save_path=None):
    """ Tests recovery vs TFT after ITS OWN forced defection. """
    strategy_names = list(strategies.keys()) # Get current names
    n_strategies = len(strategy_names)
    print(f"\nPlotting Recovery After Defection (vs TFT, sequential)...")
    cooperation_recovery_rate = np.zeros((n_strategies, recovery_rounds))
    opponent_strategy_def = tit_for_tat

    for i, name in enumerate(strategy_names):
        strategy_def = strategies[name]
        tested_strategy = strategy_def() if isinstance(strategy_def, type) else strategy_def
        opponent = opponent_strategy_def # TFT is stateless

        history1 = [COOPERATE] * lookback # Tested strategy's history (list of 0/1)
        history2 = [COOPERATE] * lookback # Opponent's history

        # Tested strategy defects once
        history1.append(DEFECT)
        # Get opponent's response (TFT sees D, responds D)
        move2 = opponent(history2.copy(), history1.copy())
        history2.append(move2)

        strategy_coop_during_recovery = []
        for r in range(recovery_rounds):
            move1 = tested_strategy(history1.copy(), history2.copy())
            history1.append(move1)
            strategy_coop_during_recovery.append(move1) # Store 0 or 1
            move2 = opponent(history2.copy(), history1.copy())
            history2.append(move2)

        if strategy_coop_during_recovery:
            # Calculate running mean (cooperation rate)
            cooperation_recovery_rate[i, :] = np.cumsum(strategy_coop_during_recovery) / np.arange(1, recovery_rounds + 1)
        else:
             cooperation_recovery_rate[i, :] = 0

    plt.figure(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, n_strategies))
    recovery_axis = np.arange(1, recovery_rounds + 1)
    for i, name in enumerate(strategy_names):
        # Use current strategy names for labels
        plt.plot(recovery_axis, cooperation_recovery_rate[i], '-', linewidth=2.5, label=name, color=colors[i])

    plt.xlabel('Rounds After Initial Defection', fontsize=14, fontweight='bold')
    plt.ylabel('Strategy\'s Cooperation Rate (During Recovery)', fontsize=14, fontweight='bold')
    plt.title(f'Strategy Recovery vs TFT After Own Defection', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='black', loc='best')
    plt.ylim(-0.05, 1.05)
    plt.axhline(y=1.0, color='green', linestyle=':', alpha=0.7)
    plt.axhline(y=0.0, color='red', linestyle=':', alpha=0.7)
    plt.tight_layout()
    if save_path:
        try: plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"Plot saved to {save_path}")
        except Exception as e: print(f"Error saving plot {save_path}: {e}")
    else: plt.show()
    plt.close()
def analyze_noise_impact(strategies_dict, noise_levels, trials_per_matchup=NUM_SIMULATIONS_PER_MATCHUP // 10, rounds_per_game=NUM_ROUNDS):
    """Analyzes strategy performance across specified noise levels sequentially."""
    strategy_names = list(strategies_dict.keys())
    results = {name: {'avg_scores': [], 'coop_rates': []} for name in strategy_names}
    actual_trials = max(10, trials_per_matchup)
    print(f"\nAnalyzing noise impact sequentially (using {actual_trials} trials per matchup)...")

    for noise in noise_levels: # Iterate through the provided levels
        print(f"  Noise level: {noise:.2f}")
        sys.stdout.flush()
        add_noise = noise > 0
        tournament_results = run_tournament(
            strategies_dict, trials_per_matchup=actual_trials,
            rounds_per_game=rounds_per_game, add_noise=add_noise, noise_level=noise
        )
        for name in strategy_names:
            if name in tournament_results:
                 results[name]['avg_scores'].append(tournament_results[name].get('avg_score', 0))
                 results[name]['coop_rates'].append(tournament_results[name].get('cooperation_rate', 0))
            else:
                 results[name]['avg_scores'].append(0); results[name]['coop_rates'].append(0)
                 print(f"Warning: Strategy {name} missing results at noise {noise:.2f}")
    return results, noise_levels

def plot_noise_impact(noise_results, noise_levels, save_path=None):
    """ Plot noise impact on strategy scores and cooperation rates. """
    strategy_names = list(noise_results.keys())
    if not strategy_names or len(noise_levels) == 0:
         print("Warning: No noise results or levels to plot.")
         return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Impact of Noise on Strategy Performance', fontsize=16, fontweight='bold')
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategy_names)))

    # --- Score vs Noise ---
    max_avg_score_overall = 1.0
    min_avg_score_overall = 0.0
    for i, name in enumerate(strategy_names):
        scores = noise_results[name]['avg_scores']
        if len(scores) != len(noise_levels):
            print(f"Warning: Mismatched data lengths for {name} in noise score plot.")
            continue
        max_avg_score_overall = max(max_avg_score_overall, max(scores) if scores else max_avg_score_overall)
        min_avg_score_overall = min(min_avg_score_overall, min(scores) if scores else min_avg_score_overall)
        ax1.plot(noise_levels, scores, 'o-', linewidth=2.5, markersize=7, label=name,
                 color=colors[i], markeredgecolor='black', markeredgewidth=0.5)
    ax1.set_xlabel('Noise Level (Probability of move flip)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Score per Game', fontsize=12, fontweight='bold')
    ax1.set_title('Strategy Performance vs. Noise', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='black', loc='best')
    # Adjust ylim dynamically
    score_padding = (max_avg_score_overall - min_avg_score_overall) * 0.05
    ax1.set_ylim(bottom=min_avg_score_overall - score_padding, top=max_avg_score_overall + score_padding)
    ax1.set_xlim(left=min(noise_levels)-0.01, right=max(noise_levels)+0.01)
    # Add vertical line at 0.5 noise (random outcome expected)
    ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.9, lw=1.5)
    ax1.text(0.51, ax1.get_ylim()[0]*0.95 + ax1.get_ylim()[1]*0.05, ' Random Play', color='gray', fontsize=9)


    # --- Coop Rate vs Noise ---
    for i, name in enumerate(strategy_names):
        coop_rates = noise_results[name]['coop_rates']
        if len(coop_rates) != len(noise_levels):
            print(f"Warning: Mismatched data lengths for {name} in noise coop plot.")
            continue
        ax2.plot(noise_levels, coop_rates, 'o-', linewidth=2.5, markersize=7, label=name,
                 color=colors[i], markeredgecolor='black', markeredgewidth=0.5)
    ax2.set_xlabel('Noise Level', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cooperation Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Cooperation Rate vs. Noise', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    # ax2.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='black', loc='best') # Legend might clutter
    ax2.set_ylim(-0.05, 1.05)
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.9, lw=1.5)
    ax2.text(max(noise_levels)*0.95, 0.52, '50%', color='gray', ha='right', fontsize=9)
    ax2.set_xlim(left=min(noise_levels)-0.01, right=max(noise_levels)+0.01)
    # Add vertical line at 0.5 noise
    ax2.axvline(x=0.5, color='gray', linestyle=':', alpha=0.9, lw=1.5)
    # ax2.text(0.51, 0.05, 'Random Play', color='gray', fontsize=9)


    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    if save_path:
        try: plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"Plot saved to {save_path}")
        except Exception as e: print(f"Error saving plot {save_path}: {e}")
    else: plt.show()
    plt.close(fig)

def plot_adaptive_response_to_cooperate_defect_ratio(strategy_def, strategy_name, opponent_cooperation_rates=None,
                                                  rounds=100, trials=10, save_path=None):
    """ Tests adaptation to opponents with fixed cooperation rates. """
    # Use the passed strategy_name for printing and titles
    print(f"\nPlotting Adaptive Response for {strategy_name} (sequential)...")
    if opponent_cooperation_rates is None:
        opponent_cooperation_rates = np.linspace(0, 1, 11)

    response_rates_mean, response_rates_std = [], []
    scores_mean, scores_std = [], []

    def fixed_cooperation_rate_strategy(rate):
        def strategy_func(history, opponent_history):
            return COOPERATE if np.random.random() < rate else DEFECT
        strategy_func.__name__ = f"FixedRate({rate:.2f})"
        return strategy_func

    for rate in opponent_cooperation_rates:
        opponent_func = fixed_cooperation_rate_strategy(rate)
        trial_responses, trial_scores = [], []
        for _ in range(trials):
            tested_strategy = strategy_def() if isinstance(strategy_def, type) else strategy_def
            result = run_game(tested_strategy, opponent_func, rounds=rounds, add_noise=False)
            trial_responses.append(result['cooperation_rate1'])
            trial_scores.append(result['score1'] / rounds) # Score per round
        response_rates_mean.append(np.mean(trial_responses))
        response_rates_std.append(np.std(trial_responses))
        scores_mean.append(np.mean(trial_scores))
        scores_std.append(np.std(trial_scores))

    response_rates_mean, response_rates_std = np.array(response_rates_mean), np.array(response_rates_std)
    scores_mean, scores_std = np.array(scores_mean), np.array(scores_std)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    # Use the passed strategy_name for the title
    fig.suptitle(f'Adaptive Response: {strategy_name} vs Fixed Cooperation Opponents', fontsize=16, fontweight='bold')
    response_color, score_color = '#3498db', '#e74c3c'

    # Cooperation response plot - use passed name for label
    ax1.errorbar(opponent_cooperation_rates, response_rates_mean, yerr=response_rates_std, fmt='o-',
               color=response_color, linewidth=2.5, markersize=7, capsize=4, label=f'{strategy_name} Response')
    ax1.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, linewidth=2, label='Ideal Linear Adaptation')
    # Use passed name for label
    ax1.set_ylabel(f'{strategy_name}\'s Cooperation Rate', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_title('Cooperation Response', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.set_ylim(-0.05, 1.05)

    # Score plot - use passed name for label
    ax2.errorbar(opponent_cooperation_rates, scores_mean, yerr=scores_std, fmt='o-',
               color=score_color, linewidth=2.5, markersize=7, capsize=4, label=f'{strategy_name} Avg. Score')
    ax2.set_xlabel('Opponent\'s Fixed Cooperation Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Score per Round', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title('Resulting Score', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    min_score, max_score = PAYOFF_MATRIX.min(), PAYOFF_MATRIX.max()
    ax2.set_ylim(min_score - 0.2, max_score + 0.2)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92, hspace=0.25)
    if save_path:
        try: plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"Plot saved to {save_path}")
        except Exception as e: print(f"Error saving plot {save_path}: {e}")
    else: plt.show()
    plt.close()

def plot_strategy_summary_dashboard(strategy_def, strategy_name, save_path=None):
    """ Dashboard showing performance against various opponents and conditions (simplified). """
    # Use passed name for printing and titles
    print(f"\nGenerating Strategy Dashboard for {strategy_name} (sequential)...")
    opponents = {'ALL_C': always_cooperate, 'ALL_D': always_defect, 'TFT': tit_for_tat,
                 'TFTT': tit_for_two_tats, 'RANDOM': random_strategy}
    # Add ForgivingRetaliator if not self-testing, for comparison
    # Use the correct class name here
    if strategy_name != 'ForgivingRetaliator' and 'ForgivingRetaliator' in globals() and isinstance(globals()['ForgivingRetaliator'], type):
         opponents['ForgivingRetaliator'] = globals()['ForgivingRetaliator']

    fig = plt.figure(figsize=(15, 8))
    # Use passed name for title
    fig.suptitle(f'Strategy Analysis Dashboard: {strategy_name}', fontsize=18, fontweight='bold')
    gs = fig.add_gridspec(2, 3, wspace=0.35, hspace=0.4)
    trials_dash = 30 # Fewer trials for dashboard plots

    # --- 1. Head-to-head vs standard ---
    ax1 = fig.add_subplot(gs[0, 0])
    opponent_names = list(opponents.keys())
    scores_vs_opp = []
    for opp_name in opponent_names:
        opp_def = opponents[opp_name]
        total_score = 0
        for _ in range(trials_dash):
            s1 = strategy_def() if isinstance(strategy_def, type) else strategy_def
            s2 = opp_def() if isinstance(opp_def, type) else opp_def
            total_score += run_game(s1, s2, rounds=NUM_ROUNDS)['score1']
        scores_vs_opp.append((total_score / trials_dash) / NUM_ROUNDS) # Avg score per round
    bar_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(opponent_names)))
    ax1.bar(opponent_names, scores_vs_opp, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)
    max_score = max(scores_vs_opp) if scores_vs_opp else 1
    for i, score in enumerate(scores_vs_opp): ax1.text(i, score + 0.05 * max_score, f'{score:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax1.set_ylabel('Avg Score per Round', fontsize=10, fontweight='bold')
    ax1.set_title('Performance vs Opponents', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(PAYOFF_MATRIX.max(), max_score * 1.1))
    ax1.tick_params(axis='x', rotation=45, labelsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # --- 2. Noise resilience (vs TFT) ---
    ax2 = fig.add_subplot(gs[0, 1])
    noise_levels = np.array([0, 0.05, 0.1, 0.15, 0.2])
    noise_scores, noise_coop_rates = [], []
    for noise in noise_levels:
        total_score, total_coop = 0, 0
        for _ in range(trials_dash):
            s1 = strategy_def() if isinstance(strategy_def, type) else strategy_def
            res = run_game(s1, tit_for_tat, rounds=NUM_ROUNDS, add_noise=(noise > 0), noise_level=noise)
            total_score += res['score1']; total_coop += res['cooperation_rate1']
        noise_scores.append((total_score / trials_dash) / NUM_ROUNDS)
        noise_coop_rates.append(total_coop / trials_dash)
    ax2.plot(noise_levels, noise_scores, 'o-', color='#e74c3c', lw=2, ms=6, label='Avg Score/Round vs TFT')
    for i, (x, y) in enumerate(zip(noise_levels, noise_scores)): ax2.text(x, y + 0.05, f'{y:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#e74c3c')
    ax2.set_xlabel('Noise Level', fontsize=10, fontweight='bold'); ax2.set_ylabel('Avg Score/Round vs TFT', fontsize=10, fontweight='bold', color='#e74c3c')
    ax2.set_title('Noise Resilience (vs TFT)', fontsize=12, fontweight='bold'); ax2.grid(True, linestyle='--', alpha=0.7); ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2b = ax2.twinx()
    # Use passed name for label
    ax2b.plot(noise_levels, noise_coop_rates, 's--', color='#2ecc71', lw=2, ms=5, label=f'{strategy_name} Coop Rate')
    ax2b.set_ylabel('Cooperation Rate', fontsize=10, fontweight='bold', color='#2ecc71'); ax2b.tick_params(axis='y', labelcolor='#2ecc71'); ax2b.set_ylim(-0.05, 1.05)
    lines, labels = ax2.get_legend_handles_labels(); lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2b.legend(lines + lines2, labels + labels2, loc='lower left', fontsize=8)

    # --- 3. Forgiveness test (Recovery vs TFT after own D) ---
    ax3 = fig.add_subplot(gs[0, 2])
    lookback_f, recovery_rounds_f = 10, 30
    history1_f, history2_f = [COOPERATE] * lookback_f, [COOPERATE] * lookback_f
    s1_f = strategy_def() if isinstance(strategy_def, type) else strategy_def
    history1_f.append(DEFECT); move2_f = tit_for_tat(history2_f.copy(), history1_f.copy()); history2_f.append(move2_f)
    recovery_coop_f = []
    for r in range(recovery_rounds_f):
        move1_f = s1_f(history1_f.copy(), history2_f.copy()); history1_f.append(move1_f); recovery_coop_f.append(move1_f)
        move2_f = tit_for_tat(history2_f.copy(), history1_f.copy()); history2_f.append(move2_f)
    recovery_rate = np.cumsum(recovery_coop_f) / np.arange(1, recovery_rounds_f + 1) if recovery_coop_f else np.zeros(recovery_rounds_f)
    ax3.plot(np.arange(1, recovery_rounds_f + 1), recovery_rate, 'o-', color='#9b59b6', lw=2, ms=5)
    ax3.set_xlabel('Rounds After Own Defection', fontsize=10, fontweight='bold'); ax3.set_ylabel('Cooperation Rate', fontsize=10, fontweight='bold')
    ax3.set_title('Forgiveness/Recovery vs TFT', fontsize=12, fontweight='bold'); ax3.grid(True, linestyle='--', alpha=0.7); ax3.set_ylim(-0.05, 1.05)
    ax3.axhline(y=1.0, color='green', linestyle=':', alpha=0.7); ax3.axhline(y=0.0, color='red', linestyle=':', alpha=0.7)

    # --- 4 & 5. Dynamics vs TFT (Score & Coop Rate) ---
    ax4 = fig.add_subplot(gs[1, 0]); ax5 = fig.add_subplot(gs[1, 1], sharex=ax4)
    # Use passed name
    dyn_tft = analyze_dynamics(strategy_def, tit_for_tat, strategy_name, 'TFT', rounds=100)
    rounds_dyn = len(dyn_tft['running_score1']); rounds_axis_dyn = np.arange(1, rounds_dyn + 1)
    # Use passed name for label
    ax4.plot(rounds_axis_dyn, dyn_tft['running_score1'], color='#3498db', lw=2, label=strategy_name)
    ax4.plot(rounds_axis_dyn, dyn_tft['running_score2'], color='#e74c3c', lw=2, label='TFT')
    ax4.set_title(f'Score Dynamics vs TFT', fontsize=12, fontweight='bold'); ax4.set_xlabel('Round', fontsize=10); ax4.set_ylabel('Cumulative Score', fontsize=10); ax4.legend(loc='upper left', fontsize=8); ax4.grid(True, linestyle='--', alpha=0.6)
    # Use passed name for label
    ax5.plot(rounds_axis_dyn, dyn_tft['running_coop1'], color='#3498db', lw=2, label=strategy_name)
    ax5.plot(rounds_axis_dyn, dyn_tft['running_coop2'], color='#e74c3c', lw=2, label='TFT')
    ax5.set_title(f'Coop. Rate Dynamics vs TFT', fontsize=12, fontweight='bold'); ax5.set_xlabel('Round', fontsize=10); ax5.set_ylabel('Running Coop. Rate', fontsize=10); ax5.legend(loc='center right', fontsize=8); ax5.grid(True, linestyle='--', alpha=0.6); ax5.set_ylim(-0.05, 1.05)

    # --- 6. Move Pattern Heatmap vs TFT ---
    ax6 = fig.add_subplot(gs[1, 2])
    history1_h, history2_h = dyn_tft['history1'], dyn_tft['history2']
    if history1_h and history2_h:
        data_h = np.array([history1_h, history2_h], dtype=int)
        cmap_h = plt.cm.get_cmap('RdYlGn', 2)
        ax6.imshow(data_h, cmap=cmap_h, aspect='auto', interpolation='nearest', vmin=0, vmax=1, extent=[-0.5, rounds_dyn - 0.5, 1.5, -0.5])
        # Use passed name for label
        ax6.set_yticks([0, 1]); ax6.set_yticklabels([strategy_name, 'TFT'], fontsize=9)
        ax6.set_xlabel('Round', fontsize=10); ax6.set_title('Move Pattern vs TFT', fontsize=12, fontweight='bold')
        ax6.set_xticks(np.arange(0, rounds_dyn, step=max(1, rounds_dyn//5))); ax6.tick_params(axis='x', labelsize=8)
    else: ax6.text(0.5, 0.5, "No data", ha='center', va='center'); ax6.set_title('Move Pattern vs TFT', fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    if save_path:
        try: plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"Plot saved to {save_path}")
        except Exception as e: print(f"Error saving plot {save_path}: {e}")
    else: plt.show()
    plt.close()

def plot_comparative_analysis(strategies, save_path=None):
    """ Comparative radar analysis (simplified metrics). """
    strategy_names = list(strategies.keys()) # Get current names
    n_strategies = len(strategy_names)
    if n_strategies == 0: return
    print(f"\nGenerating Comparative Radar Analysis (sequential)...")
    metrics = ['Avg Score', 'Coop Rate', 'vs ALL_D', 'vs TFT', 'Noise (10%)', 'Forgive']
    n_metrics = len(metrics)
    results_data = {name: np.zeros(n_metrics) for name in strategy_names}
    trials_radar = 20

    # --- Get Tournament Data (must exist globally or rerun) ---
    if 'tournament_results' not in globals() or not all(name in tournament_results for name in strategy_names):
        print("Warning: Tournament results not found or incomplete. Running quick tournament for radar...")
        global tournament_results
        tournament_results = run_tournament(strategies, trials_per_matchup=trials_radar, rounds_per_game=50) # Quick run
        if not tournament_results or not all(name in tournament_results for name in strategy_names):
             print("Error: Failed to get base tournament data for radar. Cannot generate plot.")
             return

    for i, name in enumerate(strategy_names):
        strategy_def = strategies[name]
        # Score per round
        results_data[name][0] = tournament_results[name].get('avg_score', 0) / NUM_ROUNDS
        results_data[name][1] = tournament_results[name].get('cooperation_rate', 0)
        opponents_radar = {'ALL_D': always_defect, 'TFT': tit_for_tat}
        for j, (opp_name, opp_def) in enumerate(opponents_radar.items()):
            total_score_opp = sum(run_game(strategy_def() if isinstance(strategy_def, type) else strategy_def,
                                           opp_def, rounds=50)['score1'] for _ in range(trials_radar))
            results_data[name][2 + j] = (total_score_opp / trials_radar) / 50 # Avg score per round
        total_score_noise = sum(run_game(strategy_def() if isinstance(strategy_def, type) else strategy_def,
                                          tit_for_tat, rounds=50, add_noise=True, noise_level=0.10)['score1'] for _ in range(trials_radar))
        results_data[name][4] = (total_score_noise / trials_radar) / 50 # Avg score per round
        lookback_f, recovery_rounds_f = 10, 20
        history1_f, history2_f = [COOPERATE] * lookback_f, [COOPERATE] * lookback_f
        s1_f = strategy_def() if isinstance(strategy_def, type) else strategy_def
        history1_f.append(DEFECT); move2_f = tit_for_tat(history2_f.copy(), history1_f.copy()); history2_f.append(move2_f)
        # Check if history needs copying within list comprehension
        hist1_temp = history1_f[:]
        hist2_temp = history2_f[:]
        recovery_coop_f = []
        for _ in range(recovery_rounds_f):
            move1_f = s1_f(hist1_temp.copy(), hist2_temp.copy())
            recovery_coop_f.append(move1_f)
            hist1_temp.append(move1_f)
            move2_f = tit_for_tat(hist2_temp.copy(), hist1_temp.copy())
            hist2_temp.append(move2_f)

        results_data[name][5] = np.mean(recovery_coop_f) if recovery_coop_f else 0

    # --- Normalize metrics ---
    min_vals = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 0.0]) # Approx min expected score/round
    max_vals = np.array([3.5, 1.0, 2.0, 3.5, 3.5, 1.0]) # Approx max reasonable score/round
    normalized_results = {}
    for name in strategy_names:
         clipped_vals = np.clip(results_data[name], min_vals, max_vals)
         norm_vals = (clipped_vals - min_vals) / (max_vals - min_vals + 1e-6)
         normalized_results[name] = np.clip(norm_vals, 0, 1)

    # --- Plotting ---
    cols = min(n_strategies, 3); rows = (n_strategies + cols - 1) // cols
    plot_combined = n_strategies > 1 and n_strategies <= 6 # Plot combined only for few strategies
    if plot_combined: rows += 1
    fig = plt.figure(figsize=(cols * 4.5, rows * 4.5)) # Adjusted size
    fig.suptitle('Comparative Strategy Analysis (Radar Profiles)', fontsize=16, fontweight='bold')
    # Use consistent colors based on the order in the strategies dict
    colors = plt.cm.tab10(np.linspace(0, 1, n_strategies))
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist(); angles += angles[:1]

    for i, name in enumerate(strategy_names):
        row_idx, col_idx = divmod(i, cols)
        ax = fig.add_subplot(rows, cols, i + 1, polar=True)
        values = normalized_results[name].tolist(); values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], markersize=4)
        ax.fill(angles, values, color=colors[i], alpha=0.2)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels([])
        for angle, label in zip(angles[:-1], metrics): ax.text(angle, 1.25, label, ha='center', va='center', fontsize=8, fontweight='bold')
        ax.set_ylim(0, 1); ax.set_yticks(np.arange(0.2, 1.1, 0.2)); ax.set_yticklabels([f"{t:.1f}" for t in np.arange(0.2, 1.1, 0.2)], fontsize=7, color='gray'); ax.set_rlabel_position(90)
        # Use current strategy name for title
        ax.set_title(name, fontsize=11, fontweight='bold', pad=15, color=colors[i]); ax.grid(True, linestyle='--', alpha=0.6, color='gray')

    if plot_combined:
        # Adjust placement if needed, e.g., last position might be off-center
        combined_ax_idx = rows * cols if rows > n_strategies // cols else n_strategies + 1
        if combined_ax_idx > rows * cols: # Avoid index error if grid isn't fully used
             combined_ax_idx = rows * cols
        ax_combined = fig.add_subplot(rows, cols, combined_ax_idx, polar=True)
        for i, name in enumerate(strategy_names):
            values = normalized_results[name].tolist(); values += values[:1]
            # Use current strategy name for label
            ax_combined.plot(angles, values, '-', linewidth=1.5, color=colors[i], label=name, alpha=0.9)
        ax_combined.set_xticks(angles[:-1]); ax_combined.set_xticklabels([])
        for angle, label in zip(angles[:-1], metrics): ax_combined.text(angle, 1.25, label, ha='center', va='center', fontsize=8, fontweight='bold')
        ax_combined.set_ylim(0, 1); ax_combined.set_yticks(np.arange(0.2, 1.1, 0.2)); ax_combined.set_yticklabels([])
        ax_combined.set_title('Combined Profiles', fontsize=11, fontweight='bold', pad=15); ax_combined.grid(True, linestyle='--', alpha=0.6, color='gray')
        ax_combined.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=min(n_strategies, 3), fontsize=9)

    plt.tight_layout()
    fig.subplots_adjust(top=0.90, hspace=0.6, wspace=0.4) # Adjust spacing
    if save_path:
        try: plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"Plot saved to {save_path}")
        except Exception as e: print(f"Error saving plot {save_path}: {e}")
    else: plt.show()
    plt.close()


# --- Main Execution Logic (Sequential) ---

# --- Main Execution Logic ---
def main():
    """Main function to run the simulation sequentially."""
    print(f"--- Iterated Prisoner's Dilemma Simulation ---")
    print(f"Rounds per game: {NUM_ROUNDS}")
    print(f"Trials per matchup (Tournament): {NUM_SIMULATIONS_PER_MATCHUP}")
    print(f"Default Noise level (for dynamics): {NOISE_LEVEL * 100}%")
    global tournament_results

    start_time = time.time()
    np.random.seed(42)
    # Define output directory based on strategies tested
    output_dir = 'plots_severity_punisher' # New dir name for this test
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputting plots to ./{output_dir}/")

    # Define strategies including the new one
    strategies = {
        'ALL_C': always_cooperate,
        'ALL_D': always_defect,
        'TFT': tit_for_tat,
        'TFTT': tit_for_two_tats,
        'RANDOM': random_strategy,
        'ForgivingRetaliator': ForgivingRetaliator, # Keep original FR(>=2)
        'SeverityPunisher': SeverityPunisher, # Add the new strategy
    }
    strategy_names = list(strategies.keys())
    print(f"Strategies included: {', '.join(strategy_names)}")

    print("\n--- Running Main Tournament (No Noise) ---")
    tournament_results = run_tournament(
        strategies, trials_per_matchup=NUM_SIMULATIONS_PER_MATCHUP,
        rounds_per_game=NUM_ROUNDS, add_noise=False
    )

    print("\n--- Running Noise Impact Analysis (Extended Range) ---")
    extended_noise_levels = np.linspace(0, 1, 11) # 0.0 to 1.0
    noise_results, noise_levels_used = analyze_noise_impact(
        strategies,
        noise_levels=extended_noise_levels,
        trials_per_matchup=max(10, NUM_SIMULATIONS_PER_MATCHUP // 5),
        rounds_per_game=NUM_ROUNDS
    )

    print("\n--- Analyzing Dynamics (SeverityPunisher vs TFT) ---")
    # Analyze the new strategy vs TFT
    dynamics_sp_vs_tft_no_noise = analyze_dynamics(SeverityPunisher, tit_for_tat, 'SeverityPunisher', 'TFT', noise=False)
    dynamics_sp_vs_tft_with_noise = analyze_dynamics(SeverityPunisher, tit_for_tat, 'SeverityPunisher', 'TFT', noise=True, noise_level=NOISE_LEVEL)

    print("\n--- Generating Plots ---")
    sys.stdout.flush()

    # --- Call plotting functions ---
    plot_tournament_results(tournament_results, os.path.join(output_dir, 'tournament_results.png'))
    # Plot dynamics for the new strategy
    plot_dynamics(dynamics_sp_vs_tft_no_noise, 'SeverityPunisher', 'TFT', os.path.join(output_dir, 'dynamics_SP_vs_TFT_no_noise.png'))
    plot_dynamics(dynamics_sp_vs_tft_with_noise, 'SeverityPunisher', 'TFT', os.path.join(output_dir, 'dynamics_SP_vs_TFT_with_noise.png'))
    # Plot heatmap for the new strategy
    plot_cooperation_heatmap(
        dynamics_sp_vs_tft_no_noise['history1'], dynamics_sp_vs_tft_no_noise['history2'], 'SeverityPunisher', 'TFT',
        os.path.join(output_dir, 'cooperation_heatmap_SP_vs_TFT.png')
    )
    plot_noise_impact(noise_results, noise_levels_used, os.path.join(output_dir, 'noise_impact_extended.png'))
    plot_strategy_head_to_head(strategies, trials=50, noise=False, save_path=os.path.join(output_dir, 'head_to_head_matrix_no_noise.png'))
    plot_recovery_after_defection(strategies, lookback=10, recovery_rounds=30, save_path=os.path.join(output_dir, 'recovery_after_defection.png'))
    plot_evolutionary_stability(strategies, rounds=50, population_size=50, generations=50, mutation_rate=0.05, save_path=os.path.join(output_dir, 'evolutionary_stability.png'))
    # Plot dashboards for both FR and SP
    plot_strategy_summary_dashboard(ForgivingRetaliator, 'ForgivingRetaliator', save_path=os.path.join(output_dir, 'dashboard_ForgivingRetaliator.png'))
    plot_strategy_summary_dashboard(SeverityPunisher, 'SeverityPunisher', save_path=os.path.join(output_dir, 'dashboard_SeverityPunisher.png'))
    # Plot adaptive response for both
    plot_adaptive_response_to_cooperate_defect_ratio(
        ForgivingRetaliator, 'ForgivingRetaliator',
        opponent_cooperation_rates=np.linspace(0, 1, 11), rounds=100, trials=20,
        save_path=os.path.join(output_dir, 'adaptive_response_ForgivingRetaliator.png')
    )
    plot_adaptive_response_to_cooperate_defect_ratio(
        SeverityPunisher, 'SeverityPunisher',
        opponent_cooperation_rates=np.linspace(0, 1, 11), rounds=100, trials=20,
        save_path=os.path.join(output_dir, 'adaptive_response_SeverityPunisher.png')
    )
    plot_comparative_analysis(strategies, save_path=os.path.join(output_dir, 'comparative_radar_analysis.png'))


    end_time = time.time()
    print(f"\n--- Simulation Complete ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    return {'tournament_results': tournament_results, 'noise_results': noise_results}


if __name__ == "__main__":
    main()