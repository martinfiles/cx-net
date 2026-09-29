# Post de LinkedIn — borrador

**Antes de publicar:** sube el repositorio (`git push`; ya apunta a github.com/martinfiles/cx-net), comprueba
que el enlace funciona, y elige UNA imagen (recomendada: `docs/figures/fig3.png` o `fig6.png`; versión
inglesa en `docs/figures/en/`). Idealmente, publica el preprint/Zenodo antes y enlaza ese DOI.

---

## Versión en español (recomendada)

¿Puede una red neuronal descubrir sola qué sinapsis son excitadoras y cuáles inhibidoras, viendo solo el cableado?

Lo probé con el complejo central de *Drosophila* (conectoma male-CNS, 152 neuronas): entrené una red con la topología real, dejando libre el signo de cada conexión, y comparé lo aprendido con el neurotransmisor anotado.

Resultado: **no lo recupera.** Y lo más útil del proyecto fue entender por qué:

🔹 Mi primera tarea era trivial: un decodificador constante superaba a los modelos entrenados.
🔹 Con ganancias por tipo celular sí aparece una solución que integra el rumbo, pero no es específica de la química real: 28 de 64 patrones de signo por tipo también la alcanzan (la real, puesto 14).
🔹 Al cambiar de grafo, la solución depende de las sinapsis PEN↔PEN del neuropilo noduli: sin ellas 0/3 integran, con ellas 3/3.
🔹 Parte de mis conclusiones tuvieron que retractarse tras tres rondas de revisión externa (incluida la de una tarea que fallaba por la velocidad, no por la receta).

Lo publico como informe metodológico y negativo, con el cuaderno de laboratorio completo, las retractaciones y un explorador interactivo de los 64 patrones.

Lo que me llevo: en modelado con conectoma, un buen control nulo y una revisión adversarial valen más que un resultado bonito.

Código, datos derivados y cuaderno: https://github.com/martinfiles/cx-net

¿Trabajas en modelos de dirección de cabeza o en neurociencia con conectomas? Me encantaría que rompieras esto.

#Neurociencia #Conectómica #Drosophila #MachineLearning #CienciaAbierta #Reproducibilidad

---

## Versión en inglés

Can a neural network figure out which synapses are excitatory and which inhibitory from wiring alone?

I tested it on the *Drosophila* central complex (male-CNS connectome, 152 neurons): a network with the real topology and a free sign on every connection, compared against the annotated neurotransmitter.

Result: **it doesn't recover it.** And the useful part was understanding why:

🔹 My first task was trivial: a constant decoder beat the trained models.
🔹 With per-cell-type gains a heading-integrating solution exists, but it isn't specific to the real chemistry: 28 of 64 per-type sign patterns reach it too (the real one ranks 14th).
🔹 The solution depends on PEN↔PEN synapses in the noduli: without them 0/3 seeds integrate, with them 3/3.
🔹 Several of my conclusions had to be retracted over three rounds of external review (including a task that failed because of velocity, not the training recipe).

I'm publishing it as a methodological, negative-result report, with the full lab notebook, the retractions and an interactive explorer of the 64 patterns.

Takeaway: in connectome-constrained modelling, a good null control and adversarial review are worth more than a pretty result.

Code, derived data and notebook: https://github.com/martinfiles/cx-net

Working on head-direction models or connectome neuroscience? I'd love you to break this.

#Neuroscience #Connectomics #Drosophila #MachineLearning #OpenScience #Reproducibility

---

## Consejos de publicación

- Publica martes-jueves por la mañana; pon el enlace en el primer comentario si el alcance baja con enlaces en el cuerpo.
- Etiqueta (con su permiso) a quien te haya revisado; agradécelo en el post.
- Si publicas la web personal o el explorador (`docs/explorer.html` en GitHub Pages), añade ese enlace también.
- No afirmes «hallazgo sobre la mosca»: es un resultado metodológico y negativo.
