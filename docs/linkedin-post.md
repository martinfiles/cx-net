# Post de LinkedIn — borrador

**Antes de publicar:** haz `git push` del repositorio (hay commits locales sin subir; ya apunta a github.com/martinfiles/cx-net), comprueba
que el enlace funciona, y elige UNA imagen (recomendada: `docs/figures/fig3.png` o `fig6.png`; versión
inglesa en `docs/figures/en/`). Idealmente, publica el preprint/Zenodo antes y enlaza ese DOI.

---

## Versión en español (recomendada)

Pasé semanas intentando que una IA adivinara qué conexiones de un cerebro excitan y cuáles frenan. No lo consiguió. Y lo que aprendí por el camino vale más que si hubiera funcionado.

La idea: tenemos el mapa completo de cableado del cerebro de la mosca de la fruta (el conectoma). Cada conexión es un cable, pero el mapa no dice si empuja o frena. ¿Puede una red neuronal deducirlo sola, solo con la forma del cableado y una tarea que resolver (orientarse, como una brújula interna)?

Respuesta corta: **no.** Respuesta útil: por qué.

🔹 Mi primera tarea era tan fácil que un decodificador sin cerebro ganaba a todos los modelos entrenados.
🔹 Cuando la arreglé, aparece una solución que sí se orienta... pero no es especial: probé las 64 combinaciones posibles de "quién excita y quién inhibe" por tipo de neurona y 37 también consiguen orientarse (varias, mejor que la real). La real queda en el puesto 30 de 64: una más entre muchas.
🔹 Lo repetí con una versión corregida de la tarea (y 189 entrenamientos) y la conclusión aguantó. Eso es lo que quiero que se vea: lo comprobé contra mí mismo.
🔹 Hubo hallazgos que tuve que retractar tras tres rondas de revisión externa. Están todos documentados.

Por qué importa fuera de la neurociencia: si usas IA para "descubrir" estructura en un sistema (biología, finanzas, ingeniería), un buen control nulo te dice más que un resultado bonito. Aquí el control nulo fue probar todas las alternativas.

Lo publico como informe metodológico y negativo: código, cuaderno de laboratorio con los errores incluidos y un explorador interactivo de los 64 patrones.

👉 https://github.com/martinfiles/cx-net

Si trabajas con conectomas, modelos de dirección de cabeza o simplemente con evaluación rigurosa de modelos: me encantaría que lo rompieras.

#Neurociencia #Conectómica #Drosophila #MachineLearning #CienciaAbierta #Reproducibilidad

---

## Versión en inglés

I spent weeks trying to get an AI to work out which connections in a brain excite and which inhibit. It didn't. What I learned on the way is worth more than if it had worked.

The idea: we have the complete wiring map of the fruit-fly brain (the connectome). Every connection is a wire, but the map doesn't say whether it pushes or brakes. Can a neural network infer that on its own, from the wiring shape plus one task (keeping track of heading, like an internal compass)?

Short answer: **no.** Useful answer: why.

🔹 My first task was so easy that a brainless constant decoder beat every trained model.
🔹 Once fixed, a solution that does track heading appears... but it isn't special: I tried all 64 combinations of "which cell type excites, which inhibits", and 37 also manage to track heading (several better than the real one). The real assignment ranks 30th of 64: one among many.
🔹 I repeated it on a corrected version of the task (189 training runs) and the conclusion held. That's the point: I checked it against myself.
🔹 Some findings had to be retracted after three rounds of external review. All documented.

Why it matters beyond neuroscience: if you use AI to "discover" structure in a system (biology, finance, engineering), a good null control tells you more than a pretty result. Here the null control was trying every alternative.

I'm publishing it as a methodological, negative-result report: code, a lab notebook with the mistakes included, and an interactive explorer of the 64 patterns.

👉 https://github.com/martinfiles/cx-net

If you work on connectomes, head-direction models, or just rigorous model evaluation: I'd love you to break this.

#Neuroscience #Connectomics #Drosophila #MachineLearning #OpenScience #Reproducibility

---

## Consejos de publicación

- Publica martes-jueves por la mañana; pon el enlace en el primer comentario si el alcance baja con enlaces en el cuerpo.
- Etiqueta (con su permiso) a quien te haya revisado; agradécelo en el post.
- Si publicas la web personal o el explorador (`docs/explorer.html` en GitHub Pages), añade ese enlace también.
- No afirmes «hallazgo sobre la mosca»: es un resultado metodológico y negativo.
