use core::hash::{Hash, Hasher};

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub struct Prehashed(pub u64);

impl Prehashed {
    #[must_use]
    pub fn new<T: Hash + Eq>(value: T) -> Self {
        let mut hasher = ahash::AHasher::default();
        value.hash(&mut hasher);
        Self(hasher.finish())
    }

    pub fn from_hash(hash: u64) -> Self {
        Self(hash)
    }
}

impl Default for Prehashed {
    fn default() -> Self {
        Self(0)
    }
}

impl Hash for Prehashed {
    fn hash<S: Hasher>(&self, state: &mut S) {
        self.0.hash(state)
    }
}

pub struct Prehasher {
    hash: u64,
}

impl Prehasher {
    /// Constructs a new PreHasher from a already made hashed u64.
    ///
    /// # Notes
    ///
    #[must_use]
    pub fn new(hash: u64) -> Self {
        Self { hash }
    }
}

impl Default for Prehasher {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Hasher for Prehasher {
    fn write(&mut self, _bytes: &[u8]) {
        panic!("unsupported operation");
    }

    fn write_u64(&mut self, i: u64) {
        self.hash = i;
    }

    fn finish(&self) -> u64 {
        self.hash
    }
}
