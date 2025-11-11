//! High-Performance Data Pipeline (Polars Implementation)
//!
//! This file implements the data conversion pipeline using `polars` and `ahash`.
//! It reads Parquet/CSV files and builds the three sparse matrices (`X`, `U`, `T`)
//! required by the FEASE model.

use ahash::AHashMap;
use anyhow::{Result, anyhow};
use polars::prelude::*;
use sprs::{CsMat, TriMat};

// --- 1. ID and Feature Mappings ---

/// A simple helper map for dynamic feature creation.
type FeatureMap = AHashMap<String, usize>;

/// Manages all mappings required for the model.
#[derive(Debug, Clone)]
pub struct Mappings {
    /// (User-Item) `anonymous_id` -> user_idx
    pub user_mapping: AHashMap<String, usize>,
    /// (User-Item) `view_media_id` / `media_guid` -> item_idx
    pub item_mapping: AHashMap<String, usize>,
    /// (User-Feature) "feature_name_value" (e.g., "device_Mobile") -> user_feat_idx
    pub user_feature_mapping: AHashMap<String, usize>,
    /// (Item-Feature) "feature_name_value" (e.g., "media_type_episode") -> item_feat_idx
    pub item_feature_mapping: AHashMap<String, usize>,

    // Counters for dynamically building feature maps
    pub user_feature_count: usize,
    pub item_feature_count: usize,
}

impl Mappings {
    fn new() -> Self {
        Mappings {
            user_mapping: AHashMap::new(),
            item_mapping: AHashMap::new(),
            user_feature_mapping: AHashMap::new(),
            item_feature_mapping: AHashMap::new(),
            user_feature_count: 0,
            item_feature_count: 0,
        }
    }
}

// --- 2. The Main Pipeline Function ---

/// Loads data from files and builds the three sparse matrices.
pub fn build_matrices(
    engagement_path: &str,
    metadata_path: &str,
) -> Result<(CsMat<f64>, CsMat<f64>, CsMat<f64>, Mappings)> {
    // --- 1. Load Data ---
    println!("Loading dataframes from disk...");
    let df_eng = load_dataframe(engagement_path)?;
    let df_meta = load_dataframe(metadata_path)?;

    // --- 2. Build ID Mappings ---
    println!("Building ID mappings...");
    let mut mappings = Mappings::new();

    // Get all unique users from the engagement table
    let users_ca = df_eng.column("anonymous_id")?.str()?;
    for user_id in users_ca.into_no_null_iter() {
        if !user_id.is_empty() && !mappings.user_mapping.contains_key(user_id) {
            mappings
                .user_mapping
                .insert(user_id.to_string(), mappings.user_mapping.len());
        }
    }

    // Get all unique items from BOTH tables
    let items_eng_ca = df_eng.column("view_media_id")?.str()?;
    for item_id in items_eng_ca.into_iter().flatten() {
        // flatten skips nulls
        if !item_id.is_empty() && !mappings.item_mapping.contains_key(item_id) {
            mappings
                .item_mapping
                .insert(item_id.to_string(), mappings.item_mapping.len());
        }
    }

    // Cast metadata media_guid to string if it isn't already
    let items_meta_series = df_meta.column("media_guid")?.cast(&DataType::String)?;
    let items_meta_ca = items_meta_series.str()?;
    for item_id in items_meta_ca.into_iter().flatten() {
        // flatten skips nulls
        if !item_id.is_empty() && !mappings.item_mapping.contains_key(item_id) {
            mappings
                .item_mapping
                .insert(item_id.to_string(), mappings.item_mapping.len());
        }
    }

    let num_users = mappings.user_mapping.len();
    let num_items = mappings.item_mapping.len();

    if num_users == 0 || num_items == 0 {
        return Err(anyhow!("No users or items found. Check data."));
    }

    println!(
        "Found {} unique users and {} unique items.",
        num_users, num_items
    );

    // --- 3. Initialize Triplet Matrices ---
    // We build in Triplet format first, then convert to CSR/CSC.
    let mut x_triplets = TriMat::new((num_users, num_items));
    let mut u_triplets = TriMat::new((num_users, 0)); // Cols will be set later
    let mut t_triplets = TriMat::new((0, num_items)); // Rows will be set later

    // --- 4. Process Engagement Data (Builds X and U) ---
    println!("Processing Engagement data (for X and U matrices)...");

    // Select and validate all columns we need
    let users = df_eng.column("anonymous_id")?.str()?;
    let items = df_eng.column("view_media_id")?.str()?;
    let watch_time = df_eng.column("view_seconds_watched")?.f64()?;

    let devices = df_eng.column("view_context_device_type")?.str()?;
    let plans = df_eng.column("view_subscription_plan")?.str()?;
    let countries_view = df_eng.column("view_country_code_view")?.str()?;
    let countries_acct = df_eng.column("account_country_code_account")?.str()?;
    let tenures = df_eng.column("account_tenure_days")?.i32()?; // i32 for tenure
    let regions = df_eng.column("region_major_account")?.str()?;
    let sub_statuses = df_eng.column("subscription_status")?.str()?;

    // Iterate over rows
    for i in 0..df_eng.height() {
        let user_id = match users.get(i) {
            Some(id) if !id.is_empty() => id,
            _ => continue, // Skip row if user_id is null or empty
        };

        let user_idx = match mappings.user_mapping.get(user_id) {
            Some(&idx) => idx,
            None => continue, // Should not happen, but good to be safe
        };

        // --- A: Build X Matrix (Interactions) ---
        if let (Some(item_id), Some(seconds)) = (items.get(i), watch_time.get(i)) {
            if !item_id.is_empty() && seconds > 0.0 {
                if let Some(&item_idx) = mappings.item_mapping.get(item_id) {
                    // Apply log transform: log(1 + seconds)
                    let val = (1.0 + seconds).log10();
                    x_triplets.add_triplet(user_idx, item_idx, val);
                }
            }
        }

        // --- B: Build U Matrix (User Features) ---
        // Add single-value categorical features
        add_user_feature(
            devices.get(i),
            "device",
            &mut mappings.user_feature_mapping,
            &mut u_triplets,
            user_idx,
        );
        add_user_feature(
            plans.get(i),
            "plan",
            &mut mappings.user_feature_mapping,
            &mut u_triplets,
            user_idx,
        );
        add_user_feature(
            countries_view.get(i),
            "country_view",
            &mut mappings.user_feature_mapping,
            &mut u_triplets,
            user_idx,
        );
        add_user_feature(
            countries_acct.get(i),
            "country_acct",
            &mut mappings.user_feature_mapping,
            &mut u_triplets,
            user_idx,
        );
        add_user_feature(
            regions.get(i),
            "region",
            &mut mappings.user_feature_mapping,
            &mut u_triplets,
            user_idx,
        );
        add_user_feature(
            sub_statuses.get(i),
            "sub_status",
            &mut mappings.user_feature_mapping,
            &mut u_triplets,
            user_idx,
        );

        // Add bucketized numeric feature
        if let Some(days) = tenures.get(i) {
            let tenure_bucket = bucketize_tenure(days);
            add_user_feature(
                Some(tenure_bucket),
                "tenure",
                &mut mappings.user_feature_mapping,
                &mut u_triplets,
                user_idx,
            );
        }
    }

    // --- 5. Process Metadata Data (Builds T) ---
    println!("Processing Metadata data (for T matrix)...");

    // Select and validate all columns we need
    let items_meta = items_meta_series.str()?; // Re-use from mapping step

    let genres = df_meta.column("media_genres")?.str()?;
    let tags = df_meta.column("media_tags")?.str()?;
    let media_types = df_meta.column("media_type")?.str()?;
    let audio_langs = df_meta.column("media_audio_language")?.str()?;
    let series_titles = df_meta.column("media_series_title")?.str()?;
    let primary_genres = df_meta.column("airtable_primary_genre")?.str()?;
    let secondary_genres = df_meta.column("airtable_secondary_genres")?.str()?;

    // Iterate over rows
    for i in 0..df_meta.height() {
        let item_id = match items_meta.get(i) {
            Some(id) if !id.is_empty() => id,
            _ => continue, // Skip row if item_id is null or empty
        };

        let item_idx = match mappings.item_mapping.get(item_id) {
            Some(&idx) => idx,
            None => continue, // Item exists in metadata but not in interactions
        };

        // --- C: Build T Matrix (Item Features) ---

        // Add single-value categorical features
        add_item_feature(
            media_types.get(i),
            "media_type",
            &mut mappings.item_feature_mapping,
            &mut t_triplets,
            item_idx,
        );
        add_item_feature(
            audio_langs.get(i),
            "audio_lang",
            &mut mappings.item_feature_mapping,
            &mut t_triplets,
            item_idx,
        );
        add_item_feature(
            series_titles.get(i),
            "series",
            &mut mappings.item_feature_mapping,
            &mut t_triplets,
            item_idx,
        );
        add_item_feature(
            primary_genres.get(i),
            "genre_primary",
            &mut mappings.item_feature_mapping,
            &mut t_triplets,
            item_idx,
        );

        // Add split-value categorical features
        add_split_item_features(
            genres.get(i),
            "genre",
            &mut mappings.item_feature_mapping,
            &mut t_triplets,
            item_idx,
        );
        add_split_item_features(
            tags.get(i),
            "tag",
            &mut mappings.item_feature_mapping,
            &mut t_triplets,
            item_idx,
        );
        add_split_item_features(
            secondary_genres.get(i),
            "genre_secondary",
            &mut mappings.item_feature_mapping,
            &mut t_triplets,
            item_idx,
        );
    }

    // --- 6. Finalize Matrices ---
    println!("Finalizing sparse matrices...");
    let num_user_features = mappings.user_feature_mapping.len();
    let num_item_features = mappings.item_feature_mapping.len();

    // Resize triplet matrices to their final dimensions before conversion
    u_triplets.set_dims((num_users, num_user_features));
    t_triplets.set_dims((num_item_features, num_items));

    let x_mat_csr = x_triplets.to_csr();
    let u_mat_csr = u_triplets.to_csr();
    let t_mat_csr = t_triplets.to_csr();

    println!(
        "Build complete: X(N:{}xM:{}), U(N:{}xK:{}), T(L:{}xM:{})",
        x_mat_csr.rows(),
        x_mat_csr.cols(),
        u_mat_csr.rows(),
        u_mat_csr.cols(),
        t_mat_csr.rows(),
        t_mat_csr.cols()
    );

    Ok((x_mat_csr, u_mat_csr, t_mat_csr, mappings))
}

// --- 3. Helper Functions ---

/// Reads a Parquet or CSV file from disk into a Polars DataFrame.
fn load_dataframe(path: &str) -> Result<DataFrame> {
    let file_path = std::path::Path::new(path);
    match file_path.extension().and_then(|s| s.to_str()) {
        Some("parquet") => ParquetReader::from_path(path)?
            .finish()
            .map_err(|e| e.into()),
        Some("csv") => CsvReader::from_path(path)?
            .infer_schema(Some(100))
            .has_header(true)
            .finish()
            .map_err(|e| e.into()),
        _ => Err(anyhow!(
            "Unsupported file type. Please use .parquet or .csv"
        )),
    }
}

/// Adds a single categorical feature to the User-Feature matrix.
fn add_user_feature(
    value: Option<&str>,
    prefix: &str,
    feature_map: &mut FeatureMap,
    triplets: &mut TriMat<f64>,
    user_idx: usize,
) {
    if let Some(val) = value {
        if !val.is_empty() && val != "null" {
            let feature_name = format!("{}_{}", prefix, val);
            let feature_idx = *feature_map
                .entry(feature_name)
                .or_insert_with_key(|_| feature_map.len());
            triplets.add_triplet(user_idx, feature_idx, 1.0);
        }
    }
}

/// Adds a single categorical feature to the Item-Feature matrix.
fn add_item_feature(
    value: Option<&str>,
    prefix: &str,
    feature_map: &mut FeatureMap,
    triplets: &mut TriMat<f64>,
    item_idx: usize,
) {
    if let Some(val) = value {
        if !val.is_empty() && val != "null" {
            let feature_name = format!("{}_{}", prefix, val);
            let feature_idx = *feature_map
                .entry(feature_name)
                .or_insert_with_key(|_| feature_map.len());
            // Note: T matrix is (L x M), so triplet is (feature_idx, item_idx)
            triplets.add_triplet(feature_idx, item_idx, 1.0);
        }
    }
}

/// Splits a comma-separated string (e.g., "Action,Drama") and adds each
/// part as a feature to the Item-Feature matrix.
fn add_split_item_features(
    value: Option<&str>,
    prefix: &str,
    feature_map: &mut FeatureMap,
    triplets: &mut TriMat<f64>,
    item_idx: usize,
) {
    if let Some(val_str) = value {
        if !val_str.is_empty() && val_str != "null" {
            val_str
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .for_each(|val| {
                    add_item_feature(Some(val), prefix, feature_map, triplets, item_idx);
                });
        }
    }
}

/// Converts tenure in days (integer) to a string-based bucket.
fn bucketize_tenure(days: i32) -> &'static str {
    match days {
        d if d <= 0 => "0d",
        d if d <= 7 => "1-7d",
        d if d <= 30 => "8-30d",
        d if d <= 90 => "31-90d",
        d if d <= 180 => "91-180d",
        d if d <= 365 => "181-365d",
        _ => "365d+",
    }
}

// --- Unit Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucketize_tenure() {
        assert_eq!(bucketize_tenure(0), "0d");
        assert_eq!(bucketize_tenure(-5), "0d");
        assert_eq!(bucketize_tenure(7), "1-7d");
        assert_eq!(bucketize_tenure(30), "8-30d");
        assert_eq!(bucketize_tenure(31), "31-90d");
        assert_eq!(bucketize_tenure(100), "91-180d");
        assert_eq!(bucketize_tenure(200), "181-365d");
        assert_eq!(bucketize_tenure(500), "365d+");
    }

    #[test]
    fn test_add_user_feature() {
        let mut feature_map = AHashMap::new();
        let mut triplets = TriMat::new((1, 0));
        let user_idx = 0;

        // Add a new feature
        add_user_feature(
            Some("Mobile"),
            "device",
            &mut feature_map,
            &mut triplets,
            user_idx,
        );
        assert_eq!(feature_map.len(), 1);
        assert_eq!(*feature_map.get("device_Mobile").unwrap(), 0);

        // Add an existing feature
        add_user_feature(
            Some("Mobile"),
            "device",
            &mut feature_map,
            &mut triplets,
            user_idx,
        );
        assert_eq!(feature_map.len(), 1); // Count shouldn't change

        // Add another new feature
        add_user_feature(
            Some("Premium"),
            "plan",
            &mut feature_map,
            &mut triplets,
            user_idx,
        );
        assert_eq!(feature_map.len(), 2);
        assert_eq!(*feature_map.get("plan_Premium").unwrap(), 1);

        // Add null/empty
        add_user_feature(None, "device", &mut feature_map, &mut triplets, user_idx);
        add_user_feature(
            Some(""),
            "device",
            &mut feature_map,
            &mut triplets,
            user_idx,
        );
        add_user_feature(
            Some("null"),
            "device",
            &mut feature_map,
            &mut triplets,
            user_idx,
        );
        assert_eq!(feature_map.len(), 2); // Count shouldn't change

        // Check triplets
        triplets.set_dims((1, 2));
        let mat = triplets.to_csr();
        assert_eq!(mat.get(0, 0).unwrap(), &1.0); // device_Mobile
        assert_eq!(mat.get(0, 1).unwrap(), &1.0); // plan_Premium
    }

    #[test]
    fn test_add_split_item_features() {
        let mut feature_map = AHashMap::new();
        let mut triplets = TriMat::new((0, 1));
        let item_idx = 0;

        let genres = Some("Action, Drama, ,Comedy,null"); // Test spacing, empty, null
        add_split_item_features(genres, "genre", &mut feature_map, &mut triplets, item_idx);

        assert_eq!(feature_map.len(), 3);
        assert!(feature_map.contains_key("genre_Action"));
        assert!(feature_map.contains_key("genre_Drama"));
        assert!(feature_map.contains_key("genre_Comedy"));
        assert!(!feature_map.contains_key("genre_null"));

        // Check triplets
        triplets.set_dims((3, 1));
        let mat = triplets.to_csr();
        assert_eq!(
            mat.get(*feature_map.get("genre_Action").unwrap(), 0)
                .unwrap(),
            &1.0
        );
        assert_eq!(
            mat.get(*feature_map.get("genre_Drama").unwrap(), 0)
                .unwrap(),
            &1.0
        );
        assert_eq!(
            mat.get(*feature_map.get("genre_Comedy").unwrap(), 0)
                .unwrap(),
            &1.0
        );
    }
}
