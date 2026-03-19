---
name: documentation
description: Guía para la documentación del proyecto SRI en formato LNCS. Usa este skill cuando necesites crear o actualizar la documentación de un corte, escribir secciones técnicas del informe, estructurar la justificación del modelo LSI, o preparar la documentación para defensa. También aplica para el manual de usuario del Corte 3.
---

# Documentación del SRI — Formato LNCS

## Formato obligatorio

La documentación usa la plantilla **LNCS (Lecture Notes in Computer Science)** de Springer. Se entrega en **PDF** en el repositorio.

Plantilla LaTeX: [https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines](https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines)

## Estructura del documento por corte

### Corte 1

1. **Definición del dominio**: Salud y medicina — justificar por qué es adecuado para un SRI.
2. **Modelo de recuperación**: Descripción formal del LSI con fuentes bibliográficas. Incluir:
   - Definición matemática (TF-IDF → SVD → espacio latente)
   - Justificación de por qué LSI es adecuado para el dominio médico (sinonimia médica: cefalea/dolor de cabeza, hipertensión/presión arterial alta)
   - Fuente bibliográfica de la definición (Deerwester et al., 1990 — paper original de LSI)
3. **Arquitectura del sistema**: Pipeline + Microkernel, incluir los diagramas.
4. **Estadísticas del corpus**: cantidad de documentos, tamaño promedio, fuentes, diversidad.
5. **Fuentes bibliográficas**.

### Corte 2 (acumula sobre Corte 1)

6. **Mejoras al retriever**: Qué se refinó y por qué.
7. **Módulo RAG**: Cómo se integra el recuperador con el generador.
8. **Módulo de búsqueda web**: Cuándo se activa, cómo se integran los resultados.
9. **Módulos opcionales implementados**: Justificación de la selección en relación al dominio.

### Corte 3 (acumula sobre Corte 2)

10. **Interfaz visual**: Diseño, decisiones de posicionamiento de información.
11. **Módulo de posicionamiento**: Estrategias de ranking implementadas.
12. **Manual de usuario**: Guía sencilla de uso del sistema.
13. **Opinión crítica**: Bondades e insuficiencias del sistema.
14. **Deficiencias detectadas**: Lista con posibles soluciones.

## Fuente bibliográfica del LSI

```bibtex
@article{deerwester1990indexing,
  title={Indexing by latent semantic analysis},
  author={Deerwester, Scott and Dumais, Susan T and Furnas, George W 
          and Landauer, Thomas K and Harshman, Richard},
  journal={Journal of the American Society for Information Science},
  volume={41},
  number={6},
  pages={391--407},
  year={1990}
}
```

## Tips para la documentación

- Lenguaje técnico, conciso, académico. No coloquial.
- Los diagramas se incluyen como figuras con caption y referencia en el texto.
- Cada decisión arquitectónica debe tener una justificación ("se eligió X porque Y").
- Las métricas de evaluación (Corte 3) se presentan en tablas con descripción.
- Cada corte ACTUALIZA y EXPANDE la documentación previa — no se reescribe desde cero.

## Estructura de archivos de documentación

```
docs/
├── main.tex              # Documento principal LNCS
├── references.bib        # Bibliografía
├── figures/              # Diagramas e imágenes
│   ├── architecture.png
│   ├── pipeline_flow.png
│   └── folder_structure.png
└── build.sh              # Script para compilar el PDF
```
