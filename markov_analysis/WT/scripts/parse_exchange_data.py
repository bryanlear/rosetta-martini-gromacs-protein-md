#!/usr/bin/env python3
"""Parse HREMC exchange log and checkpoints into structured CSV files for analysis."""
import re
import json
import csv
import os
from pathlib import Path

BASEDIR = Path("/home/rs199473101_SCN5A/extended_simulation/monte_carlo/wt")
OUTDIR = BASEDIR / "analysis_data"
OUTDIR.mkdir(exist_ok=True)

# ── 1. Parse per-cycle exchange data from hremc.log ──
print("Parsing exchange log...")
log_file = BASEDIR / "logs" / "hremc.log"

exchanges = []  # per-pair exchange data
permutations = []  # permutation at each reporting cycle
current_cycle = None

with open(log_file) as f:
    for line in f:
        # Match cycle line
        m = re.search(r'Cycle\s+(\d+)\s+\|.*Accept:\s+([\d.]+)%\s+\|.*Elapsed:\s+([\d.]+)s', line)
        if m:
            current_cycle = int(m.group(1))
            continue
        
        # Match pair exchange data
        m = re.search(r'Pair \((\d+),(\d+)\): V_go_i=([-\d.]+) V_go_j=([-\d.]+) ΔΔU=([-\d.]+) P=([-\d.]+) (\w+)', line)
        if m:
            exchanges.append({
                'cycle': current_cycle if current_cycle else 0,
                'pair_i': int(m.group(1)),
                'pair_j': int(m.group(2)),
                'V_go_i': float(m.group(3)),
                'V_go_j': float(m.group(4)),
                'ddU': float(m.group(5)),
                'prob': float(m.group(6)),
                'result': m.group(7)
            })
            continue
        
        # Match permutation
        m = re.search(r'Permutation: \[([^\]]+)\]', line)
        if m:
            perm = [int(x.strip()) for x in m.group(1).split(',')]
            permutations.append({
                'cycle': current_cycle,
                'permutation': perm
            })
            continue
        
        # Match pair rates
        m = re.search(r'Pair rates: (.+)', line)
        if m:
            rates_str = m.group(1)
            rates = [r.strip().rstrip('%') for r in rates_str.split('|')]
            # Store with current_cycle from previous Cycle line
            continue

# Write exchange data CSV
exchange_csv = OUTDIR / "exchange_data.csv"
with open(exchange_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['cycle', 'pair_i', 'pair_j', 'V_go_i', 'V_go_j', 'ddU', 'prob', 'result'])
    writer.writeheader()
    writer.writerows(exchanges)
print(f"  Wrote {len(exchanges)} exchange records to {exchange_csv}")

# Write permutation data CSV (replica→state mapping over time)
perm_csv = OUTDIR / "permutation_history.csv"
with open(perm_csv, 'w', newline='') as f:
    header = ['cycle'] + [f'replica_{i:02d}_state' for i in range(12)]
    writer = csv.writer(f)
    writer.writerow(header)
    for p in permutations:
        row = [p['cycle']] + p['permutation']
        writer.writerow(row)
print(f"  Wrote {len(permutations)} permutation snapshots to {perm_csv}")

# ── 2. Parse checkpoint JSONs for cumulative statistics ──
print("Parsing checkpoints...")
ckpt_dir = BASEDIR / "checkpoints"
ckpt_files = sorted(ckpt_dir.glob("checkpoint_*.json"))

ckpt_data = []
for cf in ckpt_files:
    with open(cf) as f:
        d = json.load(f)
    
    # Extract cumulative acceptance rates per pair
    total_att = d.get('total_exchanges_attempted', 0)
    total_acc = d.get('total_exchanges_accepted', 0)
    
    row = {
        'cycle': d['cycle'],
        'total_attempted': total_att,
        'total_accepted': total_acc,
        'overall_rate': total_acc / total_att * 100 if total_att > 0 else 0,
    }
    
    # Per-pair stats
    pair_att = d.get('pair_attempts', {})
    pair_acc = d.get('pair_accepts', {})
    for i in range(11):
        key = f"{i}-{i+1}"
        att = pair_att.get(key, 0)
        acc = pair_acc.get(key, 0)
        row[f'pair_{i}_{i+1}_att'] = att
        row[f'pair_{i}_{i+1}_acc'] = acc
        row[f'pair_{i}_{i+1}_rate'] = acc / att * 100 if att > 0 else 0
    
    # Replica states (lambda assignments)
    for rs in d.get('replica_states', []):
        idx = rs['index']
        row[f'rep_{idx:02d}_state'] = rs['state_index']
        row[f'rep_{idx:02d}_lambda'] = rs['lam']
        row[f'rep_{idx:02d}_energy'] = rs['potential_energy']
    
    ckpt_data.append(row)

ckpt_csv = OUTDIR / "checkpoint_summary.csv"
if ckpt_data:
    with open(ckpt_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ckpt_data[0].keys())
        writer.writeheader()
        writer.writerows(ckpt_data)
    print(f"  Wrote {len(ckpt_data)} checkpoint records to {ckpt_csv}")

# ── 3. Create lambda schedule reference ──
lambdas = ckpt_data[-1] if ckpt_data else {}
lambda_csv = OUTDIR / "lambda_values.csv"
with open(lambda_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['state_index', 'lambda'])
    for i in range(12):
        lam = lambdas.get(f'rep_{i:02d}_lambda', 'N/A')
        writer.writerow([i, lam])
print(f"  Wrote lambda schedule to {lambda_csv}")

# ── 4. Summary statistics ──
print("\n=== WT HREMC Summary ===")
if ckpt_data:
    final = ckpt_data[-1]
    print(f"Final cycle: {final['cycle']}")
    print(f"Overall acceptance: {final['overall_rate']:.1f}%")
    print(f"Total exchanges attempted: {final['total_attempted']}")
    print(f"Total exchanges accepted: {final['total_accepted']}")
    print("\nPer-pair acceptance rates:")
    for i in range(11):
        rate = final.get(f'pair_{i}_{i+1}_rate', 0)
        att = final.get(f'pair_{i}_{i+1}_att', 0)
        acc = final.get(f'pair_{i}_{i+1}_acc', 0)
        print(f"  Pair ({i},{i+1}): {rate:5.1f}%  ({acc}/{att})")

print(f"\nExchange log covers {len(exchanges)} pair evaluations")
print(f"Permutation snapshots: {len(permutations)} (every 10 cycles)")
print(f"\nAnalysis files written to: {OUTDIR}")
