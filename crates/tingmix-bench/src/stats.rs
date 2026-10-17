use serde::Serialize;

#[derive(Debug, Clone, Serialize, Default, PartialEq)]
pub struct Dist {
    pub n: usize,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
    pub mean: f64,
}

/// Nearest-rank percentiles over a copy of `v`.
pub fn dist(v: &[f64]) -> Dist {
    if v.is_empty() {
        return Dist::default();
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let at = |q: f64| s[(((q / 100.0) * s.len() as f64).ceil() as usize).clamp(1, s.len()) - 1];
    Dist {
        n: s.len(),
        p50: at(50.0),
        p95: at(95.0),
        p99: at(99.0),
        max: *s.last().unwrap(),
        mean: s.iter().sum::<f64>() / s.len() as f64,
    }
}

/// Least-squares slope of y over x (units of y per unit of x).
pub fn slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if xs.len() < 2 {
        return 0.0;
    }
    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let num: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

pub fn median(v: &[f64]) -> f64 {
    dist(v).p50
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_and_slope() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        let d = dist(&v);
        assert_eq!((d.p50, d.p95, d.p99, d.max), (50.0, 95.0, 99.0, 100.0));
        assert_eq!(dist(&[]), Dist::default());
        assert!((slope(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0]) - 2.0).abs() < 1e-12);
    }
}
