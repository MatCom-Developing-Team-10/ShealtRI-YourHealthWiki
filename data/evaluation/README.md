# Colección de evaluación (test collection)

Dataset etiquetado para evaluar la calidad de recuperación del SRI con las
métricas objetivas de la **Conferencia 4** (Evaluación, retroalimentación y
expansión de consulta).

## Archivos

- `eval_queries.json` — lista de consultas de evaluación:
  ```json
  [{"query_id": "q01", "text": "¿qué es la insulina ...?"}, ...]
  ```
- `eval_qrels.json` — juicios de relevancia (qrels) por consulta:
  ```json
  {"q01": {"<doc_id>": 2, "<doc_id>": 1}, ...}
  ```

## Convención de relevancia graduada

Sigue el esquema usado por NDCG en la conferencia:

| Grado | Significado        |
|-------|--------------------|
| 2     | Altamente relevante |
| 1     | Relevante           |
| 0     | No relevante (ausente del qrels) |

## Metodología de construcción

Los juicios se generaron sobre el corpus real (`data/documents/`, ~13 000
páginas de libros de medicina y biología) usando **densidad de términos** como
proxy de relevancia:

1. Por cada consulta se definió un patrón de términos centrales del tema
   (p. ej. `insulin`, `hemoglobina`, `potencial de acción`).
2. Se contó la frecuencia del patrón en cada documento.
3. Los documentos con densidad ≥ 3 se ordenaron por frecuencia y se tomaron
   los 12 primeros.
4. El ~40 % más denso se etiquetó como grado 2; el resto como grado 1.

> **Limitación (según la Conf_4):** la propia conferencia advierte que los
> juicios de relevancia requieren *juicio de expertos*. Este dataset es un
> proxy automático suficiente para demostrar el módulo de evaluación y comparar
> configuraciones del retriever; para la defensa final conviene complementarlo
> con revisión humana.

## Uso

```bash
# Evaluar el retriever LSI puro contra esta colección
python -m modules.evaluation.service --k 10

# Con desglose por consulta
python -m modules.evaluation.service --k 10 --per-query
```
