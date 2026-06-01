# plato-embed

> Embedding utilities for PLATO tiles — random, hash-based, value projection, and learned embeddings with similarity search

## What This Does

plato-embed creates vector embeddings from tile data. It supports four embedding methods (random, hash-based, value projection, learned projection), provides cosine similarity search over an embedding index, and tracks index statistics. Embeddings are the bridge between raw tile values and the vector space where similarity operations work.

## The Key Idea

A temperature of 22.5°C and humidity of 55% live on different scales. Embeddings project them into a shared vector space where "similar" readings are close together. The simplest method (value projection) maps scalar values to high-dimensional vectors. Hash-based embeddings are deterministic and require no training. Random embeddings provide a baseline. Learned projections adapt to your data.

## Install

```bash
cargo add plato-embed
```

## Quick Start

```rust
use plato_embed::*;

let config = EmbeddingConfig::default(); // 64-dim, normalized, value projection
let embedder = Embedder::new(config);

let embedding = embedder.embed(22.5, "temp-sensor");
println!("Vector: {:?} ({}d)", embedding.vector, embedding.dimension);

// Build an index and search
let mut index = EmbeddingIndex::new(64);
index.add(embedding);
let results = index.search(&query, 5); // top-5 nearest
```

## API Reference

| Type | Description |
|---|---|
| `EmbedMethod` | `Random` / `HashBased` / `ValueProjection` / `LearnedProjection` |
| `EmbeddingConfig { dimension, normalize, method }` | Defaults: 64-dim, normalized, value projection |
| `Embedding { vector, dimension, source_id, source_type }` | A vector embedding with metadata |
| `SearchResult { embedding, score, rank }` | A search hit |
| `EmbeddingIndex` | Collection with `add()`, `search(query, k)`, `stats()`, `remove()` |
| `IndexStats { count, avg_norm, dimension }` | Index metadata |

### Embedding Methods

| Method | How | Properties |
|---|---|---|
| Random | Seeded RNG → vector | Baseline, no meaning |
| HashBased | Hash value → deterministic vector | Deterministic, no training |
| ValueProjection | Map scalar to high-dim space | Preserves value relationships |
| LearnedProjection | Trainable projection matrix | Data-adaptive |

## Testing

25 tests: embedding creation, all four methods, normalization, index insertion/search/removal, cosine similarity, index stats, high-dimensional embeddings.

## License

Apache-2.0
