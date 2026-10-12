#!/bin/bash
# 3ª revision (punto A): tarea OU con velocidad en el rango antiguo (max_av=0.04, p90 filtrada 0.055),
# T=500, tau_v=50 (43% de ensayos llegan a 1 vuelta) y perdida sobre incrementos (loss_type='increment').
cd "$(dirname "$0")/.."
run() {
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  nohup .venv/Scripts/python.exe -W ignore -u -c "
import json, torch
torch.set_num_threads(1)
from src.cx_net.train import train
r=train(n_epochs=800, lr=0.05, trials_per_step=16, patience=150, hold_prob=0.3, anchor=True, max_av=0.04,
        ring_source='eb_synapses', ring_sign=1.0, activation='tanh', type_params=True, real_sign_control=True,
        recurrent_gain=2.0, tau=10.0, in_gain=10.0, cue_gain=10.0, run_label='$1',
        T=500, velocity_process='ou', tau_v=50.0, state_noise_std=0.02, checkpoint_selection='ema',
        loss_type='increment', $2)
print('RESULT_JSON', json.dumps(r))
" > data/interim/$1.log 2>&1 &
}
for s in 0 1 2; do run "taskv3_real_seed${s}_p" "seed=$s"; done
for k in 5 6 7 8; do run "taskv3_shuf${k}_p" "seed=0, control_shuffle_seed=$k"; done
wait
