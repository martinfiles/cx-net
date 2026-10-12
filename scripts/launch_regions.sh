#!/bin/bash
# 3ª revision (punto B): reentrena la asignacion real (3 semillas, receta de la tabla tanh/ring_sign=+1)
# sobre (a) grafo completo SIN NO y (b) PB+EB+NO. Aisla si las sinapsis PEN<->PEN del NO explican el fallo de PB+EB.
cd "$(dirname "$0")/.."
run() {  # etiqueta, kwargs extra
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  nohup .venv/Scripts/python.exe -W ignore -u -c "
import json, torch
torch.set_num_threads(1)
from src.cx_net.train import train
r=train(n_epochs=600, lr=0.05, trials_per_step=16, patience=100, hold_prob=0.3, anchor=True, max_av=0.15,
        ring_source='eb_synapses', ring_sign=1.0, activation='tanh', type_params=True, real_sign_control=True,
        recurrent_gain=2.0, tau=10.0, in_gain=10.0, cue_gain=10.0, run_label='$1', $2)
print('RESULT_JSON', json.dumps(r))
" > data/interim/$1.log 2>&1 &
}
for s in 0 1 2; do
  run "noNO_real_seed${s}_p" "seed=$s, graph_data_dir='data/raw/malecns_no_NO'"
  run "pbebNO_real_seed${s}_p" "seed=$s, graph_data_dir='data/raw/malecns_pb_eb_no'"
done
wait
