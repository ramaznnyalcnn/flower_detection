# Log — wiki

> Append-only timeline. Every LLM operation leaves an entry here.
>
> Format: `## [YYYY-MM-DD] <op> | <title>` followed by an optional detail line.
> Valid ops: `ingest`, `query`, `lint`, `create`, `update`, `delete`, `note`.
>
> Grep the last 10 entries: `grep "^## \[" log.md | tail -10`

## [2026-05-11] note | Vault initialized
Topic: **Flower Detection Project - Pattern Recognition vs Deep Learning**. Layers created: `raw/`, `wiki/{entities,concepts,sources,comparisons,synthesis}`.
Schema loader: `WIKI_GUIDE.md` + `AGENTS.md` + `.cursorrules`.

## [2026-05-11] ingest | Proje README ve mevcut kod tabanı
Kaynak: `README.md` + `clean_data.py` + `download_oxford102.py` + `download_google_images.py` + `requirements.txt`.
Oluşturulan sayfalar (10): `sources/project-readme`, `entities/{oxford-flowers-102, web-collected-images}`, `concepts/{data-cleaning-pipeline, feature-extraction, classical-classifiers, deep-learning-approach, dimensionality-reduction, evaluation-metrics}`, `synthesis/{project-overview, roadmap}`.
`index.md` güncellendi.
