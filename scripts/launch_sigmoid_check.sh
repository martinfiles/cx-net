#!/bin/bash
# Comprobacion de realizabilidad con tasas NO negativas (sigmoide): signos reales +-1 y 4 barajados.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
run() {  # etiqueta, ring_sign, extra
  nohup .venv/Scripts/python.exe -W ignore -u -c "
import json, torch
torch.set_num_threads(1)
from src.cx_net.train import train
r=train(n_epochs=600, lr=0.05, trials_per_step=16, patience=100, seed=0, hold_prob=0.3, anchor=True, max_av=0.15,
        ring_source='eb_synapses', ring_sign=$2, activation='sigmoid', type_params=True, real_sign_control=True,
        recurrent_gain=2.0, tau=10.0, in_gain=10.0, cue_gain=10.0, run_label='$1'$3)
print('RESULT_JSON', json.dumps(r))
" > data/interim/$1.log 2>&1 &
}
run sigm_real_p 1.0 ""
run sigm_real_m -1.0 ""
for k in 1 2 3 4; do run sigm_shuf${k}_p 1.0 ", control_shuffle_seed=$k"; done
