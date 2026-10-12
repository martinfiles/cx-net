#!/bin/bash
# Experimento del aprendiz de signos sobre la tarea OU corregida (preregistrado en el cuaderno, 2026-10-12).
# Receta identica a launch_types_v3.sh (800 epocas, T=500, tau_v=50, max_av=0.04, state_noise_std=0.02,
# loss_type='increment', checkpoint_selection='ema'); solo cambia quien decide el signo:
#   A: 6 signos por tipo, inicio neutro N(0, 0.1)            seeds 0-9  (primario)
#   B: 6 signos por tipo, inicio binario aleatorio +-1        seeds 0-9  (primario)
#   C: un signo por arista (9.160), inicio neutro             seeds 0-5  (descriptivo)
#   D: 6 signos por tipo, inicio en la asignacion real        seeds 0-5  (control de estabilidad)
# Reanudable: salta las corridas con results_*.json ya escrito. PARALLEL = procesos simultaneos (1 hilo cada uno).
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-.venv/Scripts/python.exe}
run_one() {
  arm=${1%:*}; s=${1#*:}; label="signlearn_${arm}_s${s}"
  [ -f "data/interim/results_${label}.json" ] && return 0
  case $arm in
    A) extra="type_sign_learner=True, type_sign_init='neutral'";;
    B) extra="type_sign_learner=True, type_sign_init='random_binary'";;
    C) extra="";;
    D) extra="type_sign_learner=True, type_sign_init='real'";;
  esac
  $PY -W ignore -u -c "
import json, torch
torch.set_num_threads(1)
from src.cx_net.train import train
r=train(n_epochs=${N_EPOCHS:-800}, lr=0.05, trials_per_step=16, patience=150, hold_prob=0.3, anchor=True, max_av=0.04,
        ring_source='eb_synapses', ring_sign=1.0, activation='tanh', type_params=True,
        recurrent_gain=2.0, tau=10.0, in_gain=10.0, cue_gain=10.0, run_label='$label',
        T=500, velocity_process='ou', tau_v=50.0, state_noise_std=0.02, checkpoint_selection='ema',
        loss_type='increment', seed=$s${extra:+, $extra})
print('RESULT_JSON', json.dumps(r))
" > data/interim/${label}.log 2>&1
}
export -f run_one
export PY
if [ -n "$SMOKE" ]; then run_one "$SMOKE"; exit; fi
{ for s in $(seq 0 9); do echo "A:$s"; echo "B:$s"; done
  for s in $(seq 0 5); do echo "C:$s"; echo "D:$s"; done; } | xargs -P ${PARALLEL:-16} -I{} bash -c 'run_one {}'
