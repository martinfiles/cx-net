#!/bin/bash
# 63 patrones de signo por tipo (mascara 1 = real, ya hecha). 21 en paralelo. Entrada (12) del cuaderno.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
run_mask() {
  m=$1
  .venv/Scripts/python.exe -W ignore -u -c "
import json, torch
torch.set_num_threads(1)
from src.cx_net.train import train
r=train(n_epochs=600, lr=0.05, trials_per_step=16, patience=100, seed=0, hold_prob=0.3, anchor=True, max_av=0.15,
        ring_source='eb_synapses', ring_sign=1.0, activation='tanh', type_params=True, real_sign_control=True,
        recurrent_gain=2.0, tau=10.0, in_gain=10.0, cue_gain=10.0, control_type_mask=$m, run_label='types_mask${m}_p')
print('RESULT_JSON', json.dumps(r))
" > data/interim/types_mask${m}_p.log 2>&1
}
export -f run_mask
seq 0 63 | grep -vx 1 | xargs -P 21 -I{} bash -c 'run_mask {}'
