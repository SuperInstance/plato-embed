use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbedMethod {
    Random,
    HashBased,
    ValueProjection,
    LearnedProjection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub dimension: usize,
    pub normalize: bool,
    pub method: EmbedMethod,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            dimension: 64,
            normalize: true,
            method: EmbedMethod::ValueProjection,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    pub vector: Vec<f64>,
    pub dimension: usize,
    pub source_id: Uuid,
    pub source_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub embedding: Embedding,
    pub score: f64,
    pub rank: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub count: usize,
    pub avg_norm: f64,
    pub dimension: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingIndex {
    embeddings: Vec<Embedding>,
    dimension: usize,
    id_map: HashMap<Uuid, usize>,
}

// ---------------------------------------------------------------------------
// Simple Pseudo-RNG (xoshiro256++)
// ---------------------------------------------------------------------------

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = (*state).wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn seeded_random_vec(dim: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..dim)
        .map(|_| {
            // map to [-1, 1)
            let bits = splitmix64(&mut state);
            (bits as f64 / u64::MAX as f64) * 2.0 - 1.0
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Hash helpers
// ---------------------------------------------------------------------------

fn hash_str(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

fn hash_to_vec(input: &str, dim: usize) -> Vec<f64> {
    // Deterministic: hash input + dimension index to produce each component
    (0..dim)
        .map(|i| {
            let combined = format!("{}:{}", input, i);
            let h = hash_str(&combined);
            (h as f64 / u64::MAX as f64) * 2.0 - 1.0
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Embedding impl
// ---------------------------------------------------------------------------

impl Embedding {
    pub fn from_values(values: &[f64], config: &EmbeddingConfig) -> Self {
        let raw = match config.method {
            EmbedMethod::Random => {
                let seed = Uuid::new_v4().as_u128() as u64;
                seeded_random_vec(config.dimension, seed)
            }
            EmbedMethod::HashBased => {
                // hash the byte representation of the values
                let s: String = values.iter().map(|v| v.to_string() + ",").collect();
                hash_to_vec(&s, config.dimension)
            }
            EmbedMethod::ValueProjection => {
                // project/truncate or pad to target dimension
                if values.len() >= config.dimension {
                    values[..config.dimension].to_vec()
                } else {
                    let mut v = values.to_vec();
                    v.resize(config.dimension, 0.0);
                    v
                }
            }
            EmbedMethod::LearnedProjection => {
                // placeholder: same as ValueProjection for now
                if values.len() >= config.dimension {
                    values[..config.dimension].to_vec()
                } else {
                    let mut v = values.to_vec();
                    v.resize(config.dimension, 0.0);
                    v
                }
            }
        };

        let mut emb = Embedding {
            vector: raw,
            dimension: config.dimension,
            source_id: Uuid::new_v4(),
            source_type: "values".to_string(),
        };
        if config.normalize {
            emb.normalize();
        }
        emb
    }

    pub fn from_hash(input: &str, dim: usize) -> Self {
        let vector = hash_to_vec(input, dim);
        let mut emb = Embedding {
            vector,
            dimension: dim,
            source_id: Uuid::new_v4(),
            source_type: "hash".to_string(),
        };
        emb.normalize();
        emb
    }

    pub fn normalize(&mut self) {
        let norm: f64 = self.vector.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            for v in &mut self.vector {
                *v /= norm;
            }
        }
    }

    pub fn l2_norm(&self) -> f64 {
        self.vector.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    pub fn dot_product(a: &Self, b: &Self) -> f64 {
        a.vector
            .iter()
            .zip(b.vector.iter())
            .map(|(x, y)| x * y)
            .sum()
    }

    pub fn cosine_similarity(a: &Self, b: &Self) -> f64 {
        let dot = Self::dot_product(a, b);
        let norm_a = a.l2_norm();
        let norm_b = b.l2_norm();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    pub fn euclidean(a: &Self, b: &Self) -> f64 {
        a.vector
            .iter()
            .zip(b.vector.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    pub fn manhattan(a: &Self, b: &Self) -> f64 {
        a.vector
            .iter()
            .zip(b.vector.iter())
            .map(|(x, y)| (x - y).abs())
            .sum()
    }
}

// ---------------------------------------------------------------------------
// EmbeddingIndex impl
// ---------------------------------------------------------------------------

impl EmbeddingIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            embeddings: Vec::new(),
            dimension: dim,
            id_map: HashMap::new(),
        }
    }

    pub fn add(&mut self, embedding: Embedding) {
        let idx = self.embeddings.len();
        self.id_map.insert(embedding.source_id, idx);
        self.embeddings.push(embedding);
    }

    pub fn search(&self, query: &Embedding, k: usize) -> Vec<SearchResult> {
        let mut scored: Vec<(usize, f64)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, e)| (i, Embedding::cosine_similarity(query, e)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(k)
            .enumerate()
            .map(|(rank, (idx, score))| SearchResult {
                embedding: self.embeddings[idx].clone(),
                score,
                rank: rank + 1,
            })
            .collect()
    }

    pub fn search_by_id(&self, id: Uuid, k: usize) -> Vec<SearchResult> {
        match self.id_map.get(&id) {
            Some(&idx) => {
                let query = &self.embeddings[idx];
                let mut results = self.search(query, k + 1);
                // exclude self
                results.retain(|r| r.embedding.source_id != id);
                results.truncate(k);
                // re-rank
                for (i, r) in results.iter_mut().enumerate() {
                    r.rank = i + 1;
                }
                results
            }
            None => Vec::new(),
        }
    }

    pub fn cluster(&self, threshold: f64) -> Vec<Vec<usize>> {
        let n = self.embeddings.len();
        if n == 0 {
            return Vec::new();
        }

        let mut assigned = vec![false; n];
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            if assigned[i] {
                continue;
            }
            let mut cluster = vec![i];
            assigned[i] = true;
            for j in (i + 1)..n {
                if assigned[j] {
                    continue;
                }
                let sim = Embedding::cosine_similarity(&self.embeddings[i], &self.embeddings[j]);
                if sim >= threshold {
                    cluster.push(j);
                    assigned[j] = true;
                }
            }
            clusters.push(cluster);
        }
        clusters
    }

    pub fn stats(&self) -> IndexStats {
        let count = self.embeddings.len();
        let avg_norm = if count == 0 {
            0.0
        } else {
            self.embeddings.iter().map(|e| e.l2_norm()).sum::<f64>() / count as f64
        };
        IndexStats {
            count,
            avg_norm,
            dimension: self.dimension,
        }
    }

    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Batch embed
// ---------------------------------------------------------------------------

pub fn batch_embed(values: &[Vec<f64>], config: &EmbeddingConfig) -> Vec<Embedding> {
    values.iter().map(|v| Embedding::from_values(v, config)).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn test_config(method: EmbedMethod) -> EmbeddingConfig {
        EmbeddingConfig {
            dimension: 8,
            normalize: true,
            method,
        }
    }

    // 1. Embedding creation from values — ValueProjection
    #[test]
    fn test_from_values_value_projection() {
        let cfg = test_config(EmbedMethod::ValueProjection);
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let emb = Embedding::from_values(&values, &cfg);
        assert_eq!(emb.dimension, 8);
        assert_eq!(emb.vector.len(), 8);
        // first 4 should be the normalized values
        assert!((emb.vector[0] - emb.vector[1]).abs() > 0.0); // they differ after normalization
    }

    // 2. Embedding creation from values — Random
    #[test]
    fn test_from_values_random() {
        let cfg = EmbeddingConfig {
            dimension: 16,
            normalize: true,
            method: EmbedMethod::Random,
        };
        let emb = Embedding::from_values(&[1.0, 2.0], &cfg);
        assert_eq!(emb.dimension, 16);
        assert!((emb.l2_norm() - 1.0).abs() < 1e-9);
    }

    // 3. Embedding creation from values — HashBased
    #[test]
    fn test_from_values_hash_based() {
        let cfg = test_config(EmbedMethod::HashBased);
        let emb = Embedding::from_values(&[1.0, 2.0, 3.0], &cfg);
        assert_eq!(emb.dimension, 8);
    }

    // 4. Embedding creation from values — LearnedProjection
    #[test]
    fn test_from_values_learned_projection() {
        let cfg = test_config(EmbedMethod::LearnedProjection);
        let emb = Embedding::from_values(&[1.0, 2.0], &cfg);
        assert_eq!(emb.dimension, 8);
    }

    // 5. Hash-based determinism
    #[test]
    fn test_hash_determinism() {
        let a = Embedding::from_hash("hello", 8);
        let b = Embedding::from_hash("hello", 8);
        assert_eq!(a.vector, b.vector);
    }

    // 6. Hash-based different inputs differ
    #[test]
    fn test_hash_different_inputs() {
        let a = Embedding::from_hash("hello", 8);
        let b = Embedding::from_hash("world", 8);
        assert_ne!(a.vector, b.vector);
    }

    // 7. Normalization — L2 norm = 1.0
    #[test]
    fn test_normalization() {
        let mut emb = Embedding {
            vector: vec![3.0, 4.0],
            dimension: 2,
            source_id: Uuid::new_v4(),
            source_type: "test".into(),
        };
        emb.normalize();
        assert!((emb.l2_norm() - 1.0).abs() < 1e-9);
    }

    // 8. Dot product
    #[test]
    fn test_dot_product() {
        let a = Embedding {
            vector: vec![1.0, 0.0],
            dimension: 2,
            source_id: Uuid::new_v4(),
            source_type: "test".into(),
        };
        let b = Embedding {
            vector: vec![0.0, 1.0],
            dimension: 2,
            source_id: Uuid::new_v4(),
            source_type: "test".into(),
        };
        assert!((Embedding::dot_product(&a, &b) - 0.0).abs() < 1e-9);
        assert!((Embedding::dot_product(&a, &a) - 1.0).abs() < 1e-9);
    }

    // 9. Cosine similarity
    #[test]
    fn test_cosine_similarity() {
        let a = Embedding {
            vector: vec![1.0, 0.0],
            dimension: 2,
            source_id: Uuid::new_v4(),
            source_type: "test".into(),
        };
        let b = Embedding {
            vector: vec![1.0, 0.0],
            dimension: 2,
            source_id: Uuid::new_v4(),
            source_type: "test".into(),
        };
        assert!((Embedding::cosine_similarity(&a, &b) - 1.0).abs() < 1e-9);
    }

    // 10. Euclidean distance
    #[test]
    fn test_euclidean() {
        let a = Embedding {
            vector: vec![0.0, 0.0],
            dimension: 2,
            source_id: Uuid::new_v4(),
            source_type: "test".into(),
        };
        let b = Embedding {
            vector: vec![3.0, 4.0],
            dimension: 2,
            source_id: Uuid::new_v4(),
            source_type: "test".into(),
        };
        assert!((Embedding::euclidean(&a, &b) - 5.0).abs() < 1e-9);
    }

    // 11. Manhattan distance
    #[test]
    fn test_manhattan() {
        let a = Embedding {
            vector: vec![0.0, 0.0],
            dimension: 2,
            source_id: Uuid::new_v4(),
            source_type: "test".into(),
        };
        let b = Embedding {
            vector: vec![3.0, 4.0],
            dimension: 2,
            source_id: Uuid::new_v4(),
            source_type: "test".into(),
        };
        assert!((Embedding::manhattan(&a, &b) - 7.0).abs() < 1e-9);
    }

    // 12. Index add and search
    #[test]
    fn test_index_add_and_search() {
        let mut idx = EmbeddingIndex::new(4);
        let cfg = EmbeddingConfig {
            dimension: 4,
            normalize: true,
            method: EmbedMethod::ValueProjection,
        };
        let e1 = Embedding::from_values(&[1.0, 0.0, 0.0, 0.0], &cfg);
        let e2 = Embedding::from_values(&[0.0, 1.0, 0.0, 0.0], &cfg);
        idx.add(e1.clone());
        idx.add(e2.clone());
        assert_eq!(idx.len(), 2);
        let results = idx.search(&e1, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].embedding.source_id, e1.source_id);
    }

    // 13. Search returns correct top-k
    #[test]
    fn test_search_top_k() {
        let mut idx = EmbeddingIndex::new(4);
        let cfg = EmbeddingConfig {
            dimension: 4,
            normalize: true,
            method: EmbedMethod::ValueProjection,
        };
        let query = Embedding::from_values(&[1.0, 0.0, 0.0, 0.0], &cfg);
        let similar = Embedding::from_values(&[0.9, 0.1, 0.0, 0.0], &cfg);
        let different = Embedding::from_values(&[0.0, 0.0, 0.0, 1.0], &cfg);
        idx.add(similar.clone());
        idx.add(different.clone());
        let results = idx.search(&query, 2);
        assert_eq!(results.len(), 2);
        assert!(results[0].score > results[1].score);
    }

    // 14. Search by ID
    #[test]
    fn test_search_by_id() {
        let mut idx = EmbeddingIndex::new(4);
        let cfg = EmbeddingConfig {
            dimension: 4,
            normalize: true,
            method: EmbedMethod::ValueProjection,
        };
        let e1 = Embedding::from_values(&[1.0, 0.0, 0.0, 0.0], &cfg);
        let id = e1.source_id;
        let e2 = Embedding::from_values(&[0.9, 0.1, 0.0, 0.0], &cfg);
        let e3 = Embedding::from_values(&[0.0, 0.0, 1.0, 0.0], &cfg);
        idx.add(e1);
        idx.add(e2);
        idx.add(e3);
        let results = idx.search_by_id(id, 2);
        assert!(results.iter().all(|r| r.embedding.source_id != id));
        assert!(results.len() <= 2);
    }

    // 15. Clustering groups similar embeddings
    #[test]
    fn test_clustering() {
        let mut idx = EmbeddingIndex::new(4);
        let cfg = EmbeddingConfig {
            dimension: 4,
            normalize: true,
            method: EmbedMethod::ValueProjection,
        };
        // two clusters
        idx.add(Embedding::from_values(&[1.0, 0.0, 0.0, 0.0], &cfg));
        idx.add(Embedding::from_values(&[0.9, 0.1, 0.0, 0.0], &cfg));
        idx.add(Embedding::from_values(&[0.0, 0.0, 1.0, 0.0], &cfg));
        idx.add(Embedding::from_values(&[0.0, 0.0, 0.9, 0.1], &cfg));
        let clusters = idx.cluster(0.8);
        assert!(clusters.len() >= 2);
    }

    // 16. Batch embedding
    #[test]
    fn test_batch_embed() {
        let cfg = test_config(EmbedMethod::ValueProjection);
        let inputs: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let results = batch_embed(&inputs, &cfg);
        assert_eq!(results.len(), 3);
        for emb in &results {
            assert_eq!(emb.dimension, 8);
        }
    }

    // 17. Stats accuracy
    #[test]
    fn test_stats() {
        let mut idx = EmbeddingIndex::new(4);
        assert!(idx.is_empty());
        let cfg = EmbeddingConfig {
            dimension: 4,
            normalize: true,
            method: EmbedMethod::ValueProjection,
        };
        idx.add(Embedding::from_values(&[1.0, 0.0, 0.0, 0.0], &cfg));
        idx.add(Embedding::from_values(&[0.0, 1.0, 0.0, 0.0], &cfg));
        let stats = idx.stats();
        assert_eq!(stats.count, 2);
        assert_eq!(stats.dimension, 4);
        assert!((stats.avg_norm - 1.0).abs() < 1e-9); // all normalized
    }

    // 18. Edge case: zero vector
    #[test]
    fn test_zero_vector() {
        let mut emb = Embedding {
            vector: vec![0.0, 0.0, 0.0],
            dimension: 3,
            source_id: Uuid::new_v4(),
            source_type: "test".into(),
        };
        emb.normalize();
        // should remain zero (no division by zero panic)
        assert_eq!(emb.vector, vec![0.0, 0.0, 0.0]);
    }

    // 19. Edge case: single dimension
    #[test]
    fn test_single_dimension() {
        let cfg = EmbeddingConfig {
            dimension: 1,
            normalize: true,
            method: EmbedMethod::ValueProjection,
        };
        let emb = Embedding::from_values(&[42.0], &cfg);
        assert_eq!(emb.dimension, 1);
        assert!((emb.vector[0].abs() - 1.0).abs() < 1e-9);
    }

    // 20. Edge case: empty index
    #[test]
    fn test_empty_index() {
        let idx = EmbeddingIndex::new(4);
        assert!(idx.is_empty());
        assert_eq!(idx.stats().count, 0);
        assert_eq!(idx.stats().avg_norm, 0.0);
        let clusters = idx.cluster(0.5);
        assert!(clusters.is_empty());
    }

    // 21. Edge case: identical embeddings
    #[test]
    fn test_identical_embeddings() {
        let a = Embedding {
            vector: vec![1.0, 2.0, 3.0],
            dimension: 3,
            source_id: Uuid::new_v4(),
            source_type: "test".into(),
        };
        let b = Embedding {
            vector: vec![1.0, 2.0, 3.0],
            dimension: 3,
            source_id: Uuid::new_v4(),
            source_type: "test".into(),
        };
        assert!((Embedding::euclidean(&a, &b) - 0.0).abs() < 1e-9);
        assert!((Embedding::manhattan(&a, &b) - 0.0).abs() < 1e-9);
    }

    // 22. Search ranking correctness
    #[test]
    fn test_search_ranking() {
        let mut idx = EmbeddingIndex::new(4);
        let cfg = EmbeddingConfig {
            dimension: 4,
            normalize: true,
            method: EmbedMethod::ValueProjection,
        };
        let q = Embedding::from_values(&[1.0, 0.0, 0.0, 0.0], &cfg);
        idx.add(Embedding::from_values(&[0.0, 0.0, 0.0, 1.0], &cfg));
        idx.add(Embedding::from_values(&[0.8, 0.2, 0.0, 0.0], &cfg));
        idx.add(Embedding::from_values(&[0.5, 0.5, 0.0, 0.0], &cfg));
        let results = idx.search(&q, 3);
        assert_eq!(results.len(), 3);
        // scores should be descending
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }
        // ranks should be 1, 2, 3
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.rank, i + 1);
        }
    }

    // 23. Source IDs are unique
    #[test]
    fn test_unique_source_ids() {
        let cfg = test_config(EmbedMethod::ValueProjection);
        let ids: HashSet<Uuid> = (0..100)
            .map(|_| Embedding::from_values(&[1.0], &cfg).source_id)
            .collect();
        assert_eq!(ids.len(), 100);
    }

    // 24. ValueProjection truncates correctly
    #[test]
    fn test_value_projection_truncates() {
        let cfg = EmbeddingConfig {
            dimension: 2,
            normalize: false,
            method: EmbedMethod::ValueProjection,
        };
        let emb = Embedding::from_values(&[1.0, 2.0, 3.0, 4.0, 5.0], &cfg);
        assert_eq!(emb.vector, vec![1.0, 2.0]);
    }

    // 25. ValueProjection pads correctly
    #[test]
    fn test_value_projection_pads() {
        let cfg = EmbeddingConfig {
            dimension: 4,
            normalize: false,
            method: EmbedMethod::ValueProjection,
        };
        let emb = Embedding::from_values(&[1.0], &cfg);
        assert_eq!(emb.vector, vec![1.0, 0.0, 0.0, 0.0]);
    }
}
