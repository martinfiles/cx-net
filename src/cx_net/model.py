"""
Red recurrente cuya única restricción estructural es la topología real del
subcircuito CX (qué neurona conecta con cuál). La MAGNITUD de cada peso queda
fija al número real de sinapsis (dato biológico, sin tocar); el SIGNO
(excitador +1 / inhibidor -1) es el único parámetro libre que se entrena.

Esta separación magnitud-fija / signo-libre es deliberada: si se dejara
también la magnitud libre, un signo "equivocado" podría compensarse con una
magnitud distinta y la pregunta de investigación (H1) dejaría de ser una
prueba limpia de si la topología predice la química real.
"""

import torch
import torch.nn as nn


class CXRingNetwork(nn.Module):
    def __init__(self, n_nodes: int, edge_index: torch.Tensor, synapse_weight: torch.Tensor,
                 tau: float = 5.0, dt: float = 1.0, recurrent_gain: float = 4.0):
        super().__init__()
        self.n_nodes = n_nodes
        self.tau = tau
        self.dt = dt
        self.recurrent_gain = recurrent_gain

        self.register_buffer("edge_src", edge_index[0])
        self.register_buffer("edge_dst", edge_index[1])

        # Normalización por nodo destino: la suma total de sinapsis
        # entrantes por neurona llega a ~850 (hasta ~1500) en datos reales,
        # lo que satura por completo tanh() y anula el gradiente. Se divide
        # cada peso por el total de sinapsis entrantes de su neurona destino
        # -- preserva las proporciones RELATIVAS reales entre las entradas de
        # cada neurona (el dato biológico que importa) y solo reescala la
        # magnitud absoluta para que la no linealidad tenga gradiente útil.
        magnitude = synapse_weight.abs()
        incoming_total = torch.zeros(n_nodes).index_add(0, edge_index[1], magnitude)
        incoming_total = incoming_total.clamp(min=1.0)
        normalized = magnitude / incoming_total[edge_index[1]]
        self.register_buffer("synapse_weight", normalized)

        # Signo libre por arista: se parametriza con tanh (continuo, en
        # [-1, 1]) en vez de un +1/-1 duro para que el entrenamiento por
        # descenso de gradiente sea posible. El signo "final" a efectos de
        # H1 se lee como sign(sign_param) una vez entrenado.
        self.sign_param = nn.Parameter(torch.randn(edge_index.shape[1]) * 0.1)

    def effective_weight(self) -> torch.Tensor:
        return torch.tanh(self.sign_param) * self.synapse_weight * self.recurrent_gain

    def step(self, r: torch.Tensor, ext_input: torch.Tensor) -> torch.Tensor:
        w = self.effective_weight()
        messages = w * r[self.edge_src]
        incoming = torch.zeros_like(r).index_add(0, self.edge_dst, messages)
        drive = torch.tanh(incoming + ext_input)
        dr = (-r + drive) * (self.dt / self.tau)
        return r + dr

    def forward(self, ext_input_seq: torch.Tensor) -> torch.Tensor:
        """ext_input_seq: [T, n_nodes] -> devuelve estados [T, n_nodes]."""
        T = ext_input_seq.shape[0]
        r = torch.zeros(self.n_nodes, device=ext_input_seq.device)
        states = []
        for t in range(T):
            r = self.step(r, ext_input_seq[t])
            states.append(r)
        return torch.stack(states)

    def learned_signs(self) -> torch.Tensor:
        """+1 / -1 por arista, para comparar más adelante con el neurotransmisor real."""
        with torch.no_grad():
            return torch.sign(self.sign_param)
