# plato-embed

Embedding utilities for PLATO tile similarity, clustering, and JEPA integration.

## Overview

- **Embedding** — dense vector with configurable creation methods (random, hash-based, value projection, learned projection)
- **EmbeddingIndex** — in-memory search index with cosine-similarity ranking, ID-based lookup, and clustering
- **Distance metrics** — cosine similarity, Euclidean, Manhattan, dot product
- **Batch embedding** — `batch_embed` for bulk processing

## Usage

```rust
use plato_embed::*;

let config = EmbeddingConfig::default();
let e1 = Embedding::from_values(&[1.0, 2.0, 3.0], &config);
let e2 = Embedding::from_hash("some-tile-id", 64);

let mut index = EmbeddingIndex::new(64);
index.add(e1);
index.add(e2);
let results = index.search(&query, 5);
```

## License

Apache-2.0
