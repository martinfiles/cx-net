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
                 tau: float = 5.0, dt: float = 1.0, recurrent_gain: float = 4.0,
                 activation: str = "tanh", type_ids: torch.Tensor | None = None,
                 learn_type_params: bool = False, learn_edge_gain: bool = False):
        """`activation="tanh"` (defecto, reproduce todo lo anterior): tasas en
        (-1, 1); una neurona inhibidora con tasa negativa excitaría a sus
        dianas, lo que no es fisiológico. `activation="rectified"`:
        relu(tanh(x)), tasas en [0, 1) (2026-09-19, entrada (7) del cuaderno).
        `activation="sigmoid"` (2026-09-20): sigmoide, tasas en (0, 1) con tasa
        basal 0.5 y sin ReLU muerta; tasas no negativas sin perder gradiente.

        `learn_type_params=True` (2026-09-19, entrada (8)): añade parámetros
        COMPARTIDOS POR TIPO celular, nunca por arista: una ganancia positiva
        por cada par de tipos origen->destino (exp de `log_pair_gain`), un
        sesgo por tipo y una escala global de la entrada. Compensan que la
        normalización por neurona destino borra las ganancias relativas entre
        tipos. El signo por arista sigue siendo el único parámetro de arista.
        Con el defecto (False) el modelo y sus state_dict no cambian.

        `learn_edge_gain=True` (2026-09-21, respuesta a la revisión externa): añade una ganancia
        positiva POR ARISTA (exp de `log_edge_gain`, init 0), es decir, magnitudes de peso libres
        con el signo fijo. Es la excepción deliberada al diseño «magnitud fija»: sirve para
        preguntar si la magnitud (conteo de sinapsis) es lo que impide integrar con los signos
        reales. Una arista de signo «equivocado» puede podarse llevando su ganancia a 0, así que
        esta variante NO es una prueba de H1. Con el defecto (False) nada cambia."""
        super().__init__()
        if activation not in ("tanh", "rectified", "sigmoid"):
            raise ValueError(f"activation desconocida: {activation}")
        self.activation = activation
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

        self.learn_type_params = learn_type_params
        if learn_type_params:
            if type_ids is None:
                raise ValueError("learn_type_params requiere type_ids")
            n_types = int(type_ids.max()) + 1
            self.n_types = n_types
            self.register_buffer("type_ids", type_ids.long())
            self.register_buffer("edge_pair", type_ids.long()[edge_index[0]] * n_types + type_ids.long()[edge_index[1]])
            self.log_pair_gain = nn.Parameter(torch.zeros(n_types * n_types))
            self.type_bias = nn.Parameter(torch.zeros(n_types))
            self.log_input_scale = nn.Parameter(torch.zeros(()))
        self.learn_edge_gain = learn_edge_gain
        if learn_edge_gain:
            self.log_edge_gain = nn.Parameter(torch.zeros(edge_index.shape[1]))

    def type_parameters(self) -> list:
        params = [self.log_pair_gain, self.type_bias, self.log_input_scale] if self.learn_type_params else []
        return params + ([self.log_edge_gain] if self.learn_edge_gain else [])

    def effective_weight(self) -> torch.Tensor:
        w = torch.tanh(self.sign_param) * self.synapse_weight * self.recurrent_gain
        if self.learn_type_params:
            w = w * torch.exp(torch.clamp(self.log_pair_gain, -5.0, 5.0))[self.edge_pair]
        if self.learn_edge_gain:
            w = w * torch.exp(torch.clamp(self.log_edge_gain, -5.0, 5.0))
        return w

    def step(self, r: torch.Tensor, ext_input: torch.Tensor) -> torch.Tensor:
        w = self.effective_weight()
        messages = w * r[self.edge_src]
        incoming = torch.zeros_like(r).index_add(0, self.edge_dst, messages)
        if self.learn_type_params:
            pre = incoming + ext_input * torch.exp(torch.clamp(self.log_input_scale, -5.0, 5.0)) + self.type_bias[self.type_ids]
        else:
            pre = incoming + ext_input
        drive = torch.sigmoid(pre) if self.activation == "sigmoid" else torch.tanh(pre)
        if self.activation == "rectified":
            drive = torch.relu(drive)
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

    def sign_confidence_penalty(self) -> torch.Tensor:
        """Término de regularización directa sobre el signo (no depende de la
        tarea): mínimo (0) cuando |tanh(sign_param)| -> 1 en toda arista,
        máximo (1) cuando sign_param = 0. Es 1 - tanh(x)^2, la derivada de
        tanh -- empuja cada arista hacia un signo confiado sin preferir cuál
        de los dos, así que la dirección la sigue decidiendo el gradiente de
        tarea; esto solo penaliza quedarse indeciso cerca de cero."""
        return (1 - torch.tanh(self.sign_param) ** 2).mean()
