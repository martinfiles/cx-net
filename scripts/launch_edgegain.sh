#!/bin/bash
# Magnitudes libres por arista con signos fijos (entrada 2026-09-21 del cuaderno). Misma receta que los patrones por tipo
# (600 epocas, tanh) + edge_gain=True. Real (3 semillas), 6 barajados por neurona, sin inhibicion (mask0) y solo EPG (mask2).
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
run() {  # etiqueta, kwargs extra
  nohup .venv/Scripts/python.exe -W ignore -u -c "
import json, torch
torch.set_num_threads(1)
from src.cx_net.train import train
r=train(n_epochs=600, lr=0.05, trials_per_step=16, patience=100, hold_prob=0.3, anchor=True, max_av=0.15,
        ring_source='eb_synapses', ring_sign=1.0, activation='tanh', type_params=True, real_sign_control=True,
        recurrent_gain=2.0, tau=10.0, in_gain=10.0, cue_gain=10.0, edge_gain=True, run_label='$1', $2)
print('RESULT_JSON', json.dumps(r))
" > data/interim/$1.log 2>&1 &
}
for s in 0 1 2; do run "edgegain_real_seed${s}_p" "seed=$s, control_type_mask=1"; done
for k in 1 2 3 4 5 6; do run "edgegain_shuf${k}_p" "seed=0, control_shuffle_seed=$k"; done
run "edgegain_mask0_p" "seed=0, control_type_mask=0"
run "edgegain_mask2_p" "seed=0, control_type_mask=2"
wait
