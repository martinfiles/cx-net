#!/bin/bash
# Tarea rediseñada (2026-09-28, respuesta a la revisión externa): velocidad OU
# (tau_v=30, en vez de ruido blanco), T=300 (mediana ~1.6 vueltas al anillo,
# en vez de ~60 grados), ruido de proceso (state_noise_std=0.02), seleccion de
# checkpoint por EMA en vez de "mejor lote". Validacion ACOTADA: 3 semillas
# reales + 4 barajados sobre el grafo mezclado de siempre (no el PB+EB), para
# aislar el efecto del rediseño de tarea.
cd "$(dirname "$0")/.."
run() {  # etiqueta, kwargs extra
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  nohup .venv/Scripts/python.exe -W ignore -u -c "
import json, torch
torch.set_num_threads(1)
from src.cx_net.train import train
r=train(n_epochs=600, lr=0.05, trials_per_step=16, patience=100, hold_prob=0.3, anchor=True, max_av=0.15,
        ring_source='eb_synapses', ring_sign=1.0, activation='tanh', type_params=True, real_sign_control=True,
        recurrent_gain=2.0, tau=10.0, in_gain=10.0, cue_gain=10.0, run_label='$1',
        T=300, velocity_process='ou', tau_v=30.0, state_noise_std=0.02, checkpoint_selection='ema', $2)
print('RESULT_JSON', json.dumps(r))
" > data/interim/$1.log 2>&1 &
}
for s in 0 1 2; do run "taskv2_real_seed${s}_p" "seed=$s"; done
for k in 5 6 7 8; do run "taskv2_shuf${k}_p" "seed=0, control_shuffle_seed=$k"; done
wait
