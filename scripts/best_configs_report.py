import json
import math

with open('sweep_logs/improved_sweep_full_final/all_results.json') as f:
    results = json.load(f)

out = []
for r in results:
    if 'config' in r and r.get('results'):
        rewards = [s.get('best_reward', float('-inf')) for s in r['results'] if s.get('best_reward', float('-inf')) > float('-inf') and not math.isnan(s.get('best_reward', float('-inf')))]
        if rewards:
            avg = sum(rewards) / len(rewards)
            out.append((avg, r['config']))
out.sort(reverse=True)

print("Top 10 configs by average best_reward (excluding failed runs):\n")
for a, c in out[:10]:
    print(f"{c} | avg_best_reward: {a:.2f}")
