#!/bin/bash
# Curriculo sobre la tarea rediseñada: ni 600 ni 1400 épocas convergieron con
# T=300/tau_v=30 fijos desde el inicio (docs/lab-notebook.md, 2026-09-28 (6) y
# 2026-09-29). Primeras 800 de 1400 épocas interpolan T 200->300 y tau_v 5->30
# (T=200/tau_v=5 es casi tan facil como la tarea vieja: 0.52 vueltas medianas);
# las 600 épocas restantes usan el objetivo T=300/tau_v=30 fijo. El held-out
# SIEMPRE se evalúa en el objetivo (ver train.py).
cd "$(dirname "$0")/.."
run() {  # etiqueta, kwargs extra
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  nohup .venv/Scripts/python.exe -W ignore -u -c "
import json, torch
torch.set_num_threads(1)
from src.cx_net.train import train
r=train(n_epochs=1400, lr=0.05, trials_per_step=16, patience=200, hold_prob=0.3, anchor=True, max_av=0.15,
        ring_source='eb_synapses', ring_sign=1.0, activation='tanh', type_params=True, real_sign_control=True,
        recurrent_gain=2.0, tau=10.0, in_gain=10.0, cue_gain=10.0, run_label='$1',
        T=300, velocity_process='ou', tau_v=30.0, state_noise_std=0.02, checkpoint_selection='ema',
        curriculum_epochs=800, curriculum_T_start=200, curriculum_tau_v_start=5.0, $2)
print('RESULT_JSON', json.dumps(r))
" > data/interim/$1.log 2>&1 &
}
for s in 0 1 2; do run "taskv2c_real_seed${s}_p" "seed=$s"; done
for k in 5 6 7 8; do run "taskv2c_shuf${k}_p" "seed=0, control_shuffle_seed=$k"; done
wait
