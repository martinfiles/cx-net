#!/bin/bash
# Enumeracion de los 64 patrones de signo por tipo sobre la tarea OU corregida (receta de launch_taskv3.sh).
# La mascara 1 (= asignacion real) se reutiliza de taskv3_real_seed{0,1,2}_p. 63 patrones x 3 semillas = 189 corridas,
# ordenadas por semilla (primero las 63 de la semilla 0). Reanudable: salta las corridas con results_*.json ya escrito.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
run_mask() {
  m=${1%:*}; s=${1#*:}; label="taskv3_mask${m}_s${s}_p"
  [ -f "data/interim/results_${label}.json" ] && return 0
  .venv/Scripts/python.exe -W ignore -u -c "
import json, torch
torch.set_num_threads(1)
from src.cx_net.train import train
r=train(n_epochs=${N_EPOCHS:-800}, lr=0.05, trials_per_step=16, patience=150, hold_prob=0.3, anchor=True, max_av=0.04,
        ring_source='eb_synapses', ring_sign=1.0, activation='tanh', type_params=True, real_sign_control=True,
        recurrent_gain=2.0, tau=10.0, in_gain=10.0, cue_gain=10.0, run_label='$label',
        T=500, velocity_process='ou', tau_v=50.0, state_noise_std=0.02, checkpoint_selection='ema',
        loss_type='increment', seed=$s, control_type_mask=$m)
print('RESULT_JSON', json.dumps(r))
" > data/interim/${label}.log 2>&1
}
export -f run_mask
if [ -n "$SMOKE" ]; then run_mask "$SMOKE"; exit; fi
for s in 0 1 2; do for m in $(seq 0 63 | grep -vx 1); do echo "$m:$s"; done; done | xargs -P ${PARALLEL:-14} -I{} bash -c 'run_mask {}'
