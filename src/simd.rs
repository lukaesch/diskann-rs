//! # SIMD-Accelerated Distance Functions
//!
//! Optimized distance calculations using SIMD instructions.
//! Falls back to scalar implementations when SIMD is not available.
//!
//! ## Supported Architectures
//!
//! - **x86_64**: AVX2, SSE4.1 (auto-detected at runtime)
//! - **aarch64**: NEON (always available on Apple Silicon)
//! - **Fallback**: Portable scalar implementation
//!
//! ## Performance
//!
//! SIMD acceleration provides 2-8x speedup for distance calculations:
//! - L2 (Euclidean): Process 8 floats per iteration (AVX) or 4 (SSE/NEON)
//! - Dot product: Same vectorization approach
//! - Cosine: Computed as 1 - dot(a,b) / (||a|| * ||b||)

use anndists::prelude::Distance;

/// SIMD-accelerated L2 (Euclidean squared) distance
#[derive(Clone, Copy, Debug, Default)]
pub struct SimdL2;

/// SIMD-accelerated dot product distance (for normalized vectors)
#[derive(Clone, Copy, Debug, Default)]
pub struct SimdDot;

/// SIMD-accelerated cosine distance
#[derive(Clone, Copy, Debug, Default)]
pub struct SimdCosine;

// =============================================================================
// Portable scalar implementations (fallback)
// =============================================================================

#[inline]
fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[inline]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn norm_squared_scalar(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum()
}

// =============================================================================
// x86_64 AVX2 implementations
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod x86_simd {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// Check if AVX2 is available at runtime
    #[inline]
    pub fn has_avx2() -> bool {
        is_x86_feature_detected!("avx2")
    }

    /// Check if SSE4.1 is available at runtime
    #[inline]
    pub fn has_sse41() -> bool {
        is_x86_feature_detected!("sse4.1")
    }

    /// L2 squared distance using AVX2 (8 floats at a time)
    #[target_feature(enable = "avx2")]
    #[inline]
    pub unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();

        let mut sum = _mm256_setzero_ps();
        let mut i = 0;

        // Process 8 elements at a time
        while i + 8 <= n {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
            i += 8;
        }

        // Horizontal sum of 256-bit vector
        let high = _mm256_extractf128_ps(sum, 1);
        let low = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(high, low);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let final_sum = _mm_add_ss(sums, shuf2);
        let mut result = _mm_cvtss_f32(final_sum);

        // Handle remaining elements
        while i < n {
            let d = a[i] - b[i];
            result += d * d;
            i += 1;
        }

        result
    }

    /// Dot product using AVX2
    #[target_feature(enable = "avx2")]
    #[inline]
    pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();

        let mut sum = _mm256_setzero_ps();
        let mut i = 0;

        while i + 8 <= n {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            sum = _mm256_fmadd_ps(va, vb, sum);
            i += 8;
        }

        // Horizontal sum
        let high = _mm256_extractf128_ps(sum, 1);
        let low = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(high, low);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let final_sum = _mm_add_ss(sums, shuf2);
        let mut result = _mm_cvtss_f32(final_sum);

        while i < n {
            result += a[i] * b[i];
            i += 1;
        }

        result
    }

    /// Norm squared using AVX2
    #[target_feature(enable = "avx2")]
    #[inline]
    pub unsafe fn norm_squared_avx2(a: &[f32]) -> f32 {
        let n = a.len();
        let mut sum = _mm256_setzero_ps();
        let mut i = 0;

        while i + 8 <= n {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            sum = _mm256_fmadd_ps(va, va, sum);
            i += 8;
        }

        let high = _mm256_extractf128_ps(sum, 1);
        let low = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(high, low);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let final_sum = _mm_add_ss(sums, shuf2);
        let mut result = _mm_cvtss_f32(final_sum);

        while i < n {
            result += a[i] * a[i];
            i += 1;
        }

        result
    }

    /// L2 squared using SSE4.1 (4 floats at a time)
    #[target_feature(enable = "sse4.1")]
    #[inline]
    pub unsafe fn l2_squared_sse41(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();

        let mut sum = _mm_setzero_ps();
        let mut i = 0;

        while i + 4 <= n {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            let diff = _mm_sub_ps(va, vb);
            let sq = _mm_mul_ps(diff, diff);
            sum = _mm_add_ps(sum, sq);
            i += 4;
        }

        // Horizontal sum
        let shuf = _mm_movehdup_ps(sum);
        let sums = _mm_add_ps(sum, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let final_sum = _mm_add_ss(sums, shuf2);
        let mut result = _mm_cvtss_f32(final_sum);

        while i < n {
            let d = a[i] - b[i];
            result += d * d;
            i += 1;
        }

        result
    }

    /// Dot product using SSE4.1
    #[target_feature(enable = "sse4.1")]
    #[inline]
    pub unsafe fn dot_product_sse41(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();

        let mut sum = _mm_setzero_ps();
        let mut i = 0;

        while i + 4 <= n {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            let prod = _mm_mul_ps(va, vb);
            sum = _mm_add_ps(sum, prod);
            i += 4;
        }

        let shuf = _mm_movehdup_ps(sum);
        let sums = _mm_add_ps(sum, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let final_sum = _mm_add_ss(sums, shuf2);
        let mut result = _mm_cvtss_f32(final_sum);

        while i < n {
            result += a[i] * b[i];
            i += 1;
        }

        result
    }
}

// =============================================================================
// aarch64 NEON implementations
// =============================================================================

#[cfg(target_arch = "aarch64")]
mod neon_simd {
    use std::arch::aarch64::*;

    /// L2 squared distance using NEON (4 floats at a time)
    #[inline]
    pub fn l2_squared_neon(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();

        // SAFETY: NEON is always available on aarch64
        unsafe {
            let mut sum = vdupq_n_f32(0.0);
            let mut i = 0;

            while i + 4 <= n {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                let diff = vsubq_f32(va, vb);
                sum = vfmaq_f32(sum, diff, diff);
                i += 4;
            }

            // Horizontal sum
            let mut result = vaddvq_f32(sum);

            while i < n {
                let d = a[i] - b[i];
                result += d * d;
                i += 1;
            }

            result
        }
    }

    /// Dot product using NEON
    #[inline]
    pub fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();

        // SAFETY: NEON is always available on aarch64
        unsafe {
            let mut sum = vdupq_n_f32(0.0);
            let mut i = 0;

            while i + 4 <= n {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                sum = vfmaq_f32(sum, va, vb);
                i += 4;
            }

            let mut result = vaddvq_f32(sum);

            while i < n {
                result += a[i] * b[i];
                i += 1;
            }

            result
        }
    }

    /// Norm squared using NEON
    #[inline]
    pub fn norm_squared_neon(a: &[f32]) -> f32 {
        let n = a.len();

        // SAFETY: NEON is always available on aarch64
        unsafe {
            let mut sum = vdupq_n_f32(0.0);
            let mut i = 0;

            while i + 4 <= n {
                let va = vld1q_f32(a.as_ptr().add(i));
                sum = vfmaq_f32(sum, va, va);
                i += 4;
            }

            let mut result = vaddvq_f32(sum);

            while i < n {
                result += a[i] * a[i];
                i += 1;
            }

            result
        }
    }
}

// =============================================================================
// Unified dispatch functions
// =============================================================================

/// Compute L2 squared distance with best available SIMD
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if x86_simd::has_avx2() {
            return unsafe { x86_simd::l2_squared_avx2(a, b) };
        }
        if x86_simd::has_sse41() {
            return unsafe { x86_simd::l2_squared_sse41(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return neon_simd::l2_squared_neon(a, b);
    }

    #[allow(unreachable_code)]
    l2_squared_scalar(a, b)
}

/// Compute dot product with best available SIMD
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if x86_simd::has_avx2() {
            return unsafe { x86_simd::dot_product_avx2(a, b) };
        }
        if x86_simd::has_sse41() {
            return unsafe { x86_simd::dot_product_sse41(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return neon_simd::dot_product_neon(a, b);
    }

    #[allow(unreachable_code)]
    dot_product_scalar(a, b)
}

/// Compute squared norm with best available SIMD
#[inline]
pub fn norm_squared(a: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if x86_simd::has_avx2() {
            return unsafe { x86_simd::norm_squared_avx2(a) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return neon_simd::norm_squared_neon(a);
    }

    #[allow(unreachable_code)]
    norm_squared_scalar(a)
}

/// Compute cosine distance with best available SIMD
/// Returns 1 - cosine_similarity, so 0 = identical, 2 = opposite
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = norm_squared(a).sqrt();
    let norm_b = norm_squared(b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    let cosine_sim = dot / (norm_a * norm_b);
    1.0 - cosine_sim.clamp(-1.0, 1.0)
}

// =============================================================================
// Distance trait implementations
// =============================================================================

impl Distance<f32> for SimdL2 {
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        l2_squared(a, b)
    }
}

impl Distance<f32> for SimdDot {
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        // For ANN, we want distance (lower = closer)
        // Assuming normalized vectors: distance = 1 - dot_product
        1.0 - dot_product(a, b)
    }
}

impl Distance<f32> for SimdCosine {
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        cosine_distance(a, b)
    }
}

// =============================================================================
// Runtime info
// =============================================================================

/// Returns information about SIMD capabilities
pub fn simd_info() -> SimdInfo {
    SimdInfo {
        #[cfg(target_arch = "x86_64")]
        avx2: x86_simd::has_avx2(),
        #[cfg(not(target_arch = "x86_64"))]
        avx2: false,

        #[cfg(target_arch = "x86_64")]
        sse41: x86_simd::has_sse41(),
        #[cfg(not(target_arch = "x86_64"))]
        sse41: false,

        #[cfg(target_arch = "aarch64")]
        neon: true,
        #[cfg(not(target_arch = "aarch64"))]
        neon: false,
    }
}

/// Information about available SIMD features
#[derive(Debug, Clone)]
pub struct SimdInfo {
    pub avx2: bool,
    pub sse41: bool,
    pub neon: bool,
}

impl std::fmt::Display for SimdInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut features = Vec::new();
        if self.avx2 {
            features.push("AVX2");
        }
        if self.sse41 {
            features.push("SSE4.1");
        }
        if self.neon {
            features.push("NEON");
        }
        if features.is_empty() {
            write!(f, "SIMD: none (scalar fallback)")
        } else {
            write!(f, "SIMD: {}", features.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_squared_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let expected: f32 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum();

        let result = l2_squared(&a, &b);
        assert!((result - expected).abs() < 1e-5, "expected {expected}, got {result}");
    }

    #[test]
    fn test_l2_squared_large() {
        // Test with dimension that requires multiple SIMD iterations + remainder
        let dim = 133; // Not divisible by 4 or 8
        let a: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i * 2) as f32).collect();

        let expected = l2_squared_scalar(&a, &b);
        let result = l2_squared(&a, &b);

        assert!(
            (result - expected).abs() < 1e-3,
            "expected {expected}, got {result}"
        );
    }

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let result = dot_product(&a, &b);

        assert!((result - expected).abs() < 1e-5, "expected {expected}, got {result}");
    }

    #[test]
    fn test_dot_product_large() {
        let dim = 128;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.02).collect();

        let expected = dot_product_scalar(&a, &b);
        let result = dot_product(&a, &b);

        assert!(
            (result - expected).abs() < 1e-3,
            "expected {expected}, got {result}"
        );
    }

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let result = cosine_distance(&a, &a);
        assert!(result.abs() < 1e-5, "identical vectors should have distance ~0, got {result}");
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let result = cosine_distance(&a, &b);
        assert!((result - 1.0).abs() < 1e-5, "orthogonal vectors should have distance ~1, got {result}");
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = a.iter().map(|x| -x).collect();
        let result = cosine_distance(&a, &b);
        assert!((result - 2.0).abs() < 1e-5, "opposite vectors should have distance ~2, got {result}");
    }

    #[test]
    fn test_simd_info() {
        let info = simd_info();
        println!("{}", info);
        // Just verify it doesn't panic
    }

    #[test]
    fn test_distance_trait_impl() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let l2 = SimdL2;
        let result = l2.eval(&a, &b);
        assert!(result > 0.0);

        let cosine = SimdCosine;
        let result = cosine.eval(&a, &b);
        assert!(result >= 0.0 && result <= 2.0);
    }
}
