use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VectorDType {
    #[default]
    F32,
}

impl VectorDType {
    pub fn size_bytes(self) -> usize {
        match self {
            VectorDType::F32 => 4,
        }
    }
}

/// Distance / similarity metric used when ranking vector field values.
///
/// All metrics are presented to callers in a "higher is better" orientation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    L2,
    #[default]
    Cosine,
    Dot,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct VectorOptions {
    indexed: bool,
    fast: bool,
    dim: usize,
    dtype: VectorDType,
    metric: Metric,
}

impl VectorOptions {
    /// Returns true iff the value is a fast field.
    #[inline]
    pub fn is_fast(&self) -> bool {
        self.fast
    }

    /// Returns true iff the value is indexed and therefore searchable.
    #[inline]
    pub fn is_indexed(&self) -> bool {
        self.indexed
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn dtype(&self) -> VectorDType {
        self.dtype
    }

    #[inline]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn with_dtype(mut self, dtype: VectorDType) -> VectorOptions {
        self.dtype = dtype;
        self
    }

    #[inline]
    pub fn bytes_per_vector(&self) -> usize {
        self.dim * self.dtype.size_bytes()
    }

    /// Whether this field's `(metric, dtype)` requires write-time
    /// unit-normalization (see `vector::distance::maybe_normalize_bytes`).
    /// Currently only `Cosine + F32`.
    pub fn needs_normalization(&self) -> bool {
        matches!(
            (self.metric, self.dtype),
            (Metric::Cosine, VectorDType::F32)
        )
    }

    pub fn set_indexed(mut self) -> Self {
        self.indexed = true;
        self
    }

    pub fn set_fast(mut self) -> Self {
        self.fast = true;
        self
    }

    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    // pub fn set_dim(mut self, dim: usize) -> Self {
    //     self.dim = dim;
    //     self
    // }

    pub fn with_metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    // pub fn set_metric(mut self, metric: Metric) -> Self {
    //     self.metric = metric;
    //     self
    // }
}

#[cfg(test)]
mod tests {
    use crate::schema::{Schema, VectorOptions};

    #[test]
    fn test_vector_field_schema_round_trip() {
        let mut schema_builder = Schema::builder();
        let options =  VectorOptions::default().with_dim(128);
        schema_builder.add_vector_field("embedding", options);
        let schema = schema_builder.build();

        let schema_json = serde_json::to_string_pretty(&schema).unwrap();
        let expected = r#"[
  {
    "name": "embedding",
    "type": "vector",
    "options": {
      "indexed": true,
      "fast": true,
      "dim": 128,
      "dtype": "f32",
      "metric": "cosine"
    }
  }
]"#;
        assert_eq!(schema_json, expected);

        let deserialized: Schema = serde_json::from_str(expected).unwrap();
        assert_eq!(schema, deserialized);
    }
}
