//! This file implements a high-performance Dynamic Programming (DP) solver for
//! Whole Page Optimization (WPO). It models page layout generation as a Multiple-Choice
//! Knapsack Problem (MCKP) with sequential adjacency constraints.
//!
//! Specifically, it maximizes the global page utility under a maximum pixel height
//! budget, enforcing that two banner formats are never placed consecutively.

use pyo3::prelude::*;

/// Formats in which a recommendation tray/shelf can be visually rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[pyclass(eq, eq_int)]
pub enum Format {
    None = 0,
    Carousel = 1,
    Grid2x3 = 2,
    Banner = 3,
}

impl Format {
    pub fn from_usize(val: usize) -> Self {
        match val {
            1 => Format::Carousel,
            2 => Format::Grid2x3,
            3 => Format::Banner,
            _ => Format::None,
        }
    }
}

/// A specific visual option for a tray layout.
#[derive(Debug, Clone)]
pub struct TrayOption {
    pub format: Format,
    /// Discretized rendering height units (e.g., 1 unit = 50px)
    pub height: usize,
    /// Predicted utility score (e.g., CTR * business_weight)
    pub utility: f64,
    /// Total item count displayed by this visual option
    pub item_count: usize,
}

/// A vertical tray slot containing one or more visual formatting options.
#[derive(Debug, Clone)]
pub struct Tray {
    pub id: usize,
    pub options: Vec<TrayOption>,
}

/// Represents a modular visual constraint applied to WPO layout solving.
pub trait LayoutConstraint: std::fmt::Debug + Send + Sync {
    /// Checks whether transitioning from `prev_format` to `curr_format` is valid.
    /// Used for spatial adjacency validation.
    fn is_valid_transition(
        &self,
        _prev_format: Format,
        _curr_format: Format,
        _current_tray_idx: usize,
    ) -> bool {
        true
    }

    /// Checks if a specific `format` is allowed to be placed at `tray_idx`.
    /// Used for slot-specific exclusion rules.
    fn is_valid_format_at_slot(&self, _format: Format, _tray_idx: usize) -> bool {
        true
    }

    /// Checks if the completed backtracking layout sequence satisfies global page rules.
    /// Evaluated post-DP for performance.
    fn is_valid_sequence(&self, _sequence: &[Format]) -> bool {
        true
    }
}

/// Dynamic stack-allocated rules representing concrete layout constraints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintRule {
    NoConsecutive(Format),
    DisallowedAtSlot { format: Format, slot_idx: usize },
    MaxOccurrences { format: Format, limit: usize },
}

impl LayoutConstraint for ConstraintRule {
    fn is_valid_transition(&self, prev_format: Format, curr_format: Format, _idx: usize) -> bool {
        match self {
            ConstraintRule::NoConsecutive(target) => {
                !(curr_format == *target && prev_format == *target)
            }
            _ => true,
        }
    }

    fn is_valid_format_at_slot(&self, format: Format, tray_idx: usize) -> bool {
        match self {
            ConstraintRule::DisallowedAtSlot { format: target, slot_idx } => {
                !(format == *target && tray_idx == *slot_idx)
            }
            _ => true,
        }
    }

    fn is_valid_sequence(&self, sequence: &[Format]) -> bool {
        match self {
            ConstraintRule::MaxOccurrences { format: target, limit } => {
                let count = sequence.iter().filter(|f| *f == target).count();
                count <= *limit
            }
            _ => true,
        }
    }
}

/// Solves the Tray Layout Optimization with dynamic constraints.
pub fn solve_tray_layout_with_constraints(
    trays: &[Tray],
    max_height: usize,
    constraints: &[ConstraintRule],
) -> (f64, Vec<Format>) {
    let n = trays.len();
    if n == 0 {
        return (0.0, vec![]);
    }

    let num_formats = 4; // None, Carousel, Grid2x3, Banner
    let height_stride = num_formats;
    let tray_stride = (max_height + 1) * height_stride;

    // DP Table: flat 1D vector to guarantee L1/L2 cache locality.
    // Index mapping: i * tray_stride + h * height_stride + format_idx
    let mut dp = vec![-f64::INFINITY; n * (max_height + 1) * num_formats];
    let mut parent = vec![0usize; n * (max_height + 1) * num_formats]; // For backtracking

    // Base Case: Initialize for the first tray (i = 0)
    let tray_0 = &trays[0];
    for opt in &tray_0.options {
        let f_idx = opt.format as usize;

        // Evaluate Slot-Specific Constraints for slot 0
        let mut is_slot_allowed = true;
        for constraint in constraints {
            if !constraint.is_valid_format_at_slot(opt.format, 0) {
                is_slot_allowed = false;
                break;
            }
        }
        if !is_slot_allowed {
            continue;
        }

        if opt.height <= max_height {
            let state_idx = 0 * tray_stride + opt.height * height_stride + f_idx;
            dp[state_idx] = opt.utility;
        }
    }

    // DP Transitions
    for i in 1..n {
        let tray_i = &trays[i];
        for h in 0..=max_height {
            for opt in &tray_i.options {
                let f_idx = opt.format as usize;

                // Evaluate Slot-Specific Constraints
                let mut is_slot_allowed = true;
                for constraint in constraints {
                    if !constraint.is_valid_format_at_slot(opt.format, i) {
                        is_slot_allowed = false;
                        break;
                    }
                }
                if !is_slot_allowed {
                    continue;
                }

                if h < opt.height {
                    continue;
                }
                let prev_h = h - opt.height;

                // Find the best previous format option k that is compatible
                let mut best_val = -f64::INFINITY;
                let mut best_prev_f_idx = 0;

                for k in 0..num_formats {
                    // Evaluate Transition Constraints
                    let mut is_transition_allowed = true;
                    for constraint in constraints {
                        if !constraint.is_valid_transition(Format::from_usize(k), opt.format, i) {
                            is_transition_allowed = false;
                            break;
                        }
                    }
                    if !is_transition_allowed {
                        continue;
                    }

                    let prev_state_idx = (i - 1) * tray_stride + prev_h * height_stride + k;
                    let prev_val = dp[prev_state_idx];
                    if prev_val > best_val {
                        best_val = prev_val;
                        best_prev_f_idx = k;
                    }
                }

                // If a valid path exists, update the DP state
                if best_val != -f64::INFINITY {
                    let current_state_idx = i * tray_stride + h * height_stride + f_idx;
                    dp[current_state_idx] = best_val + opt.utility;
                    parent[current_state_idx] = best_prev_f_idx;
                }
            }
        }
    }

    // Find all valid candidate states in the final tray (n - 1)
    let mut candidates = Vec::with_capacity((max_height + 1) * num_formats);
    for h in 0..=max_height {
        for f in 0..num_formats {
            let state_idx = (n - 1) * tray_stride + h * height_stride + f;
            let val = dp[state_idx];
            if val != -f64::INFINITY {
                candidates.push((val, h, f));
            }
        }
    }

    // Sort candidates by utility descending
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Backtrack from each candidate to find the first globally valid sequence
    for (val, h_start, f_start) in candidates {
        let mut selected_formats = vec![Format::None; n];
        let mut current_h = h_start;
        let mut current_f = f_start;

        for i in (0..n).rev() {
            let format = Format::from_usize(current_f);
            selected_formats[i] = format;

            if i > 0 {
                let opt = trays[i]
                    .options
                    .iter()
                    .find(|o| o.format as usize == current_f)
                    .unwrap();
                let state_idx = i * tray_stride + current_h * height_stride + current_f;
                current_f = parent[state_idx];
                current_h -= opt.height;
            }
        }

        // Post-backtracking global sequence check
        let mut is_global_valid = true;
        for constraint in constraints {
            if !constraint.is_valid_sequence(&selected_formats) {
                is_global_valid = false;
                break;
            }
        }

        if is_global_valid {
            return (val, selected_formats);
        }
    }

    (0.0, vec![Format::None; n])
}

/// Solves the Tray Layout Optimization using a cache-friendly flat DP table.
///
/// Optimizes: Maximize total utility under height constraint.
/// Adjacency constraint: Cannot have two Banners consecutively.
pub fn solve_tray_layout(trays: &[Tray], max_height: usize) -> (f64, Vec<Format>) {
    solve_tray_layout_with_constraints(
        trays,
        max_height,
        &[ConstraintRule::NoConsecutive(Format::Banner)],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_tray_layout_basic() {
        // Tray 0 options
        let t0 = Tray {
            id: 0,
            options: vec![
                TrayOption { format: Format::None, height: 0, utility: 0.0, item_count: 0 },
                TrayOption { format: Format::Carousel, height: 2, utility: 1.5, item_count: 5 },
                TrayOption { format: Format::Banner, height: 4, utility: 3.0, item_count: 1 },
            ],
        };

        // Tray 1 options
        let t1 = Tray {
            id: 1,
            options: vec![
                TrayOption { format: Format::None, height: 0, utility: 0.0, item_count: 0 },
                TrayOption { format: Format::Carousel, height: 2, utility: 2.0, item_count: 5 },
                TrayOption { format: Format::Banner, height: 4, utility: 4.5, item_count: 1 },
            ],
        };

        let trays = vec![t0, t1];

        // Scenario 1: Max Height = 8.
        // Attempting to select Banner (utility 3.0, ht 4) + Banner (utility 4.5, ht 4)
        // is prevented by the consecutive banner adjacency constraint.
        // Therefore, it should select Carousel (ht 2, util 1.5) + Banner (ht 4, util 4.5) -> total utility = 6.0
        // or Banner (ht 4, util 3.0) + Carousel (ht 2, util 2.0) -> total utility = 5.0.
        // Carousel (2) + Banner (4) is optimal.
        let (val, formats) = solve_tray_layout(&trays, 8);
        assert_eq!(val, 6.0);
        assert_eq!(formats, vec![Format::Carousel, Format::Banner]);

        // Scenario 2: Max Height = 3.
        // Height constraint is 3.
        // Possible combinations:
        // - Carousel (ht 2, util 1.5) + None (ht 0, util 0.0) -> total utility = 1.5
        // - None (ht 0, util 0.0) + Carousel (ht 2, util 2.0) -> total utility = 2.0
        // None + Carousel is optimal.
        let (val, formats) = solve_tray_layout(&trays, 3);
        assert_eq!(val, 2.0);
        assert_eq!(formats, vec![Format::None, Format::Carousel]);
    }

    #[test]
    fn test_solve_tray_layout_with_disallowed_slot() {
        let t0 = Tray {
            id: 0,
            options: vec![
                TrayOption { format: Format::None, height: 0, utility: 0.0, item_count: 0 },
                TrayOption { format: Format::Carousel, height: 2, utility: 1.5, item_count: 5 },
                TrayOption { format: Format::Banner, height: 4, utility: 3.0, item_count: 1 },
            ],
        };

        let t1 = Tray {
            id: 1,
            options: vec![
                TrayOption { format: Format::None, height: 0, utility: 0.0, item_count: 0 },
                TrayOption { format: Format::Carousel, height: 2, utility: 2.0, item_count: 5 },
                TrayOption { format: Format::Banner, height: 4, utility: 4.5, item_count: 1 },
            ],
        };

        let trays = vec![t0, t1];

        // Disallow Banner in Slot 1
        let constraints = vec![
            ConstraintRule::DisallowedAtSlot { format: Format::Banner, slot_idx: 1 }
        ];

        // Should fall back to Banner (t0) + Carousel (t1) -> total utility = 5.0 (since Banner t1 is disallowed)
        let (val, formats) = solve_tray_layout_with_constraints(&trays, 8, &constraints);
        assert_eq!(val, 5.0);
        assert_eq!(formats, vec![Format::Banner, Format::Carousel]);
    }

    #[test]
    fn test_solve_tray_layout_with_max_occurrences() {
        let t0 = Tray {
            id: 0,
            options: vec![
                TrayOption { format: Format::None, height: 0, utility: 0.0, item_count: 0 },
                TrayOption { format: Format::Carousel, height: 2, utility: 1.0, item_count: 5 },
                TrayOption { format: Format::Banner, height: 4, utility: 3.0, item_count: 1 },
            ],
        };

        let t1 = Tray {
            id: 1,
            options: vec![
                TrayOption { format: Format::None, height: 0, utility: 0.0, item_count: 0 },
                TrayOption { format: Format::Carousel, height: 2, utility: 2.0, item_count: 5 },
                TrayOption { format: Format::Banner, height: 4, utility: 4.0, item_count: 1 },
            ],
        };

        let trays = vec![t0, t1];

        // Enforce max 1 total Banner occurrence (so both Banner+Banner or Banner+Carousel are subject to check)
        // Wait, standard solver might find Carousel (1.0) + Banner (4.0) -> total utility 5.0 (which has 1 banner).
        // If we set max 0 total Banner occurrences:
        let constraints = vec![
            ConstraintRule::MaxOccurrences { format: Format::Banner, limit: 0 }
        ];

        // Should return Carousel + Carousel -> utility = 3.0
        let (val, formats) = solve_tray_layout_with_constraints(&trays, 8, &constraints);
        assert_eq!(val, 3.0);
        assert_eq!(formats, vec![Format::Carousel, Format::Carousel]);
    }
}
