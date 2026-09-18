# Discusión metodológica (borrador) — CX-Net

> Borrador de trabajo para la sección de discusión del preprint. Fuente:
> `docs/lab-notebook.md` (registro completo, 2026-09-16 a 2026-09-18). Este
> documento reorganiza esa cronología en una narrativa argumentativa; el
> cuaderno sigue siendo la fuente primaria de datos y decisiones.

## 1. Resultado central

Entrenamos una red recurrente cuya única restricción estructural es la
topología sináptica real del núcleo del sistema de dirección de cabeza del
complejo central de *Drosophila melanogaster* (152 neuronas, 9.160 aristas;
`EPG`, `EPGt`, `PEN_a(PEN1)`, `PEN_b(PEN2)`, `PEG`, `Delta7`; male-cns:v1.0).
La magnitud de cada peso queda fija al número real de sinapsis; el signo
(excitador/inhibidor) es el único parámetro libre, entrenado por descenso de
gradiente sobre una tarea de integración de rumbo, sin acceso nunca al
neurotransmisor real de cada neurona.

La red aprende a resolver la tarea de forma consistente y muy por encima del
azar en todas las variantes probadas. Sin embargo, **el signo sináptico
aprendido no coincide con el neurotransmisor real anotado más de lo
esperable por azar** (test de permutación de etiqueta de neurotransmisor,
2000 permutaciones). De todas las evaluaciones de H1 realizadas a lo largo
del proyecto, ninguna sobrevive como evidencia reproducible a favor de H1.

Este resultado negativo no depende de una sola corrida: se llegó a él tras
descartar sistemáticamente, con metodología multi-semilla, la explicación
más obvia, que la falta de señal se debiera simplemente a una potencia
estadística insuficiente por polarización de signo débil.

## 2. El problema de potencia estadística y cómo se resolvió

Un test de acuerdo de signo solo es informativo si la red se ha
comprometido con un signo por arista (`|tanh(sign_param)|` cercano a 1).
Las primeras evaluaciones de H1 (2026-09-16/17, ver cuaderno, entradas
2026-09-16 (7) y 2026-09-17 (2)/(5)) se hicieron sobre modelos con
polarización débil (`mean_abs_sign` entre 0.31 y 0.36, muy por debajo del
umbral de referencia de 0.5 usado en este proyecto). El resultado (p>0.9)
no era concluyente: un modelo que apenas decidió sus signos no permite
distinguir entre "H1 es falsa" y "el test no tuvo ocasión de detectar
nada".

Se probaron tres vías distintas para forzar más compromiso de signo, cada
una con protocolo multi-semilla (3 a 8 semillas, set de validación
held-out fijo, comparación por solapamiento entre grupos) para evitar
repetir el error metodológico temprano del proyecto: una sola corrida por
configuración (entrada 2026-09-16 (11) del cuaderno).

| Vía | Mecanismo | Mejor `mean_abs_sign` | ¿Sin solapamiento vs. baseline? |
|---|---|---|---|
| Hiperparámetros de dinámica (`tau`, `recurrent_gain`, momentum, `trials_per_step`) | Optimización | ~0.31-0.35, sin diferencia consistente | No aplica: no hubo efecto |
| `hold_prob` (rediseño de tarea: tramos de quietud forzada, memoria sin entrada) | Diseño de tarea | 0.349-0.364 | Sí, pero insuficiente para cruzar el umbral |
| `hold_prob` + `perturb_amp` (rediseño de tarea: ruido de entrada a filtrar) | Diseño de tarea | 0.465 ± 0.163 (bimodal) | No, solapa con `hold_prob` solo |
| `hold_prob` + `sign_reg` (regularización directa sobre el parámetro de signo) | Optimización, independiente de la tarea | **0.687 ± 0.123** | **Sí, en 8 de 8 semillas, por un margen amplio** |

Solo las dos últimas vías modifican de forma significativa la
polarización, y solo `sign_reg` (penalización `1 - tanh(sign_param)^2`,
que no favorece ningún signo en particular, solo penaliza la indecisión)
la resuelve de forma limpia y reproducible: 7 de 8 semillas cruzan el
umbral de referencia (`frac_polarized_gt_0.9 > 10%`), con un coste
moderado y desigual en el desempeño de la tarea (`held_out_loss` sube de
0.603±0.009 a 0.631±0.053, aunque la mayoría de las semillas quedan
indistinguibles del baseline).

## 3. Con la polarización resuelta, H1 sigue sin evidencia

Este es el punto argumentativo central de la discusión: con `sign_reg`, la
falta de polarización deja de ser una explicación viable para la ausencia
de señal. Sobre las 8 semillas de `hold_prob=0.3` + `sign_reg=0.05` (7 de
ellas muy por encima del umbral de referencia), solo 1 de 8 evaluaciones
de H1 da p<0.05 (p=0.0055), una tasa de resultado positivo indistinguible
del ~5% esperado por azar al correr 8 tests de permutación independientes.
El mismo patrón, un resultado significativo aislado que no se replica, ya
había aparecido con `perturb_amp` (1 de 8 semillas, entrada 2026-09-18 del
cuaderno) y se retractó explícitamente como evidencia de H1 tras la
réplica.

El hecho de que dos mecanismos ortogonales converjan en el mismo
resultado es una evidencia más fuerte que cualquiera de los dos por
separado. Uno actúa sobre el diseño de la tarea conductual; el otro actúa
directamente sobre el parámetro de signo, sin pasar por la tarea. Ambos
logran polarizar la red (uno de forma parcial, el otro de forma casi
completa) y en ambos casos el acuerdo con el neurotransmisor real sigue
sin superar el azar. Esa convergencia hace mucho menos plausible que la
ausencia de señal sea un artefacto del método elegido.

## 4. Interpretación

Los datos son compatibles con, al menos, tres lecturas distintas, que este
proyecto no puede distinguir entre sí con el diseño actual:

1. **La topología por sí sola no codifica información suficiente sobre
   identidad química** como para que un descenso de gradiente agnóstico a
   la tarea pueda recuperarla. El signo sináptico real podría depender de
   más contexto del que capta este subcircuito acotado (152 neuronas
   núcleo, sin los ~30 subtipos de ring neurons ni los
   PFN/PFL/hDelta/vDelta del fan-shaped body completo).
2. **La tarea conductual elegida (integración de rumbo) no impone las
   mismas restricciones funcionales** que dieron forma a la asignación
   real de neurotransmisor a lo largo de la evolución. Por bien resuelta
   que esté, una sola tarea es una ventana estrecha sobre las demandas
   funcionales reales del circuito.
3. **Limitaciones de la arquitectura**: la magnitud de cada peso queda fija
   al conteo de sinapsis (una decisión deliberada, para que el signo sea
   la única variable libre y H1 sea una prueba limpia, ver `model.py`).
   Pero esa misma decisión le quita a la red la posibilidad de compensar
   un signo equivocado con la magnitud, que es precisamente el mecanismo
   que permitiría distinguir una arista donde el signo importa de otra
   donde no.

Ninguna de las tres lecturas implica que H1 sea falsa en general. Lo que
sí implican es que **este diseño experimental concreto no tiene el poder
para confirmarla ni para refutarla de forma concluyente**, incluso después
de resolver el cuello de botella de la polarización. Ese es el hallazgo
metodológico que se reporta.

## 5. Limitaciones explícitas

- **`ring_angle` sin validar**: la posición angular usada para decodificar
  el rumbo es una aproximación secuencial (glomérulo asignado a una
  posición equiespaciada), no la tabla de correspondencia glomérulo-cuña
  real de Hulse et al. (2021, Fig. 10). Un intento de corrección
  (2026-09-16, entrada 10) empeoró la polarización y fue revertido; sigue
  sin resolverse. Esto afecta la fiabilidad cuantitativa de la
  decodificación de rumbo (no directamente el test de H1, que no depende
  de `ring_angle`), pero sí la interpretación de si la tarea se resuelve
  de forma biológicamente correcta.
- **Alcance del subcircuito**: 152 neuronas núcleo del sistema de
  dirección de cabeza, sin las ring neurons de entrada visual ni el
  fan-shaped body completo. Un circuito más amplio podría comportarse de
  forma distinta.
- **Una sola tarea conductual**: integración de rumbo, con dos variantes
  de dificultad (`hold_prob`, `perturb_amp`). No se probaron tareas
  cualitativamente distintas (navegación hacia una meta, integración
  multisensorial) por restricción de alcance y tiempo del proyecto.
- **Desglose H2 con n bajo por tipo celular**: 152 neuronas repartidas en
  6 tipos (`Delta7`=42, `EPGt`=4). Los tipos con menos neuronas tienen
  pocas aristas de origen, así que cualquier señal, o ausencia de señal,
  en ellos debe leerse con esa salvedad.
- **`sign_reg` es una intervención posterior al diseño original**: se
  añadió específicamente para resolver la duda de potencia estadística
  tras observar el techo de polarización. Es un control metodológico
  válido para esa pregunta puntual, pero no debe presentarse como parte
  del diseño experimental original.

## 6. Qué no se hizo (fuera de alcance, candidatos para trabajo futuro)

- Escalar a un subcircuito CX más amplio o al connectoma completo de
  MaleCNS v1.0.
- Probar tareas conductuales cualitativamente distintas, o varias tareas
  simultáneas, más allá de las dos variantes de dificultad probadas.
- Permitir magnitud entrenable. Esto rompería la limpieza de H1 tal como
  está planteada, pero podría usarse como análisis complementario para
  distinguir aristas sensibles al signo de aristas donde el signo no
  importa.
- Comparar directamente contra el modelo nulo que preserva el grado
  topológico, usado por Dhiman (2026). Aquí se usó permutación de la
  etiqueta de neurotransmisor, que responde una pregunta distinta (acuerdo
  de signo, no velocidad de aprendizaje).

## 7. Enunciado propuesto para la sección de discusión (borrador de texto)

> Entrenar una red cuya única variable libre es el signo sináptico, sobre
> la topología real de un circuito de navegación bien caracterizado,
> produce soluciones funcionalmente competentes para la tarea de
> integración de rumbo, sin que el signo aprendido reproduzca la identidad
> química real más allá del azar. Descartamos sistemáticamente la
> explicación más simple, que la falta de señal se debiera a una potencia
> estadística insuficiente por falta de compromiso de signo, mediante dos
> mecanismos independientes: un rediseño de la tarea conductual y una
> regularización aplicada directamente sobre el parámetro de signo. Esta
> última resuelve el problema de polarización de forma robusta
> (polarización fuerte en 7 de 8 semillas) sin alterar el resultado: el
> acuerdo con el neurotransmisor real permanece indistinguible del
> esperado por azar. Interpretamos esto como evidencia de que, al menos
> para este subcircuito y esta tarea, la topología sináptica por sí sola,
> sin información adicional sobre el repertorio funcional completo del
> circuito, no basta para que una red entrenada por gradiente recupere la
> química real de sus sinapsis.
