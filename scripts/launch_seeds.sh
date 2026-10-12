#!/bin/bash
# Replica de los patrones por tipo con semillas 1 y 2 (63 patrones; la mascara 1 ya tiene semillas 1 y 2 en dose_real_seed*).
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
run_mask() {
  m=${1%:*}; s=${1#*:}
  .venv/Scripts/python.exe -W ignore -u -c "
import json, torch
torch.set_num_threads(1)
from src.cx_net.train import train
r=train(n_epochs=600, lr=0.05, trials_per_step=16, patience=100, seed=$s, hold_prob=0.3, anchor=True, max_av=0.15,
        ring_source='eb_synapses', ring_sign=1.0, activation='tanh', type_params=True, real_sign_control=True,
        recurrent_gain=2.0, tau=10.0, in_gain=10.0, cue_gain=10.0, control_type_mask=$m, run_label='types_mask${m}_s${s}_p')
print('RESULT_JSON', json.dumps(r))
" > data/interim/types_mask${m}_s${s}_p.log 2>&1
}
export -f run_mask
# primero espera a que termine la ronda 2 de ganancias por arista
while [ "$(ls data/interim/results_edgegain_lr005_*_p.json 2>/dev/null | wc -l)" -lt 10 ]; do sleep 60; done
for s in 1 2; do for m in $(seq 0 63 | grep -vx 1); do echo "$m:$s"; done; done | xargs -P 16 -I{} bash -c 'run_mask {}'
