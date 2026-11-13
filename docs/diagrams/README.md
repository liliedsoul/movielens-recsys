# Диаграммы ПР № 2

| Файл | Диаграмма |
|---|---|
| `c4_context.dot` | C4, уровень контекста |
| `c4_container.dot` | C4, уровень контейнеров |
| `use_case.dot` (+ `actor.png`) | UML, варианты использования |
| `sequence_batch.mmd` | UML, последовательность обработки новой порции |
| `activity_pipeline.dot` | UML, деятельность: пайплайн обработки данных |

Пересборка изображений (из этого каталога; нужны Graphviz и mermaid-cli):

```bash
for f in c4_context c4_container use_case activity_pipeline; do dot -Tpng -Gdpi=170 $f.dot -o $f.png; done
mmdc -i sequence_batch.mmd -o sequence_batch.png -c mermaid_config.json -s 2.5 -b white
```
