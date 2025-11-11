//! This file defines the Rust structs that correspond to your data schemas.
//! These are primarily for type safety, documentation, and potential
//! row-based deserialization.

use serde::Deserialize;
use std::collections::HashMap;

// --- Content Metadata Structs ---

/// Represents a row in the `Content Metadata` table.
#[derive(Debug, Deserialize)]
pub struct ContentMetadataRecord {
    #[serde(rename = "media_guid")]
    pub media_guid: Option<String>,
    #[serde(rename = "media_type")]
    pub media_type: Option<String>,
    #[serde(rename = "media_order")]
    pub media_order: Option<f64>, // decimal(20,4)
    #[serde(rename = "media_audio_locale")]
    pub media_audio_locale: Option<String>,
    #[serde(rename = "media_audio_language")]
    pub media_audio_language: Option<String>,
    #[serde(rename = "media_hardsub_locale")]
    pub media_hardsub_locale: Option<String>,
    #[serde(rename = "media_hardsub_language")]
    pub media_hardsub_language: Option<String>,
    #[serde(rename = "media_duration_seconds")]
    pub media_duration_seconds: Option<f64>,
    #[serde(rename = "media_runtime")]
    pub media_runtime: Option<String>,
    #[serde(rename = "media_title")]
    pub media_title: Option<String>,
    #[serde(rename = "media_description")]
    pub media_description: Option<String>,
    #[serde(rename = "media_genres")]
    pub media_genres: Option<String>, // Comma-separated
    #[serde(rename = "media_tags")]
    pub media_tags: Option<String>, // Comma-separated
    #[serde(rename = "media_migrated_from_funimation")]
    pub media_migrated_from_funimation: Option<bool>,
    #[serde(rename = "media_air_time")]
    pub media_air_time: Option<String>, // timestamp
    #[serde(rename = "media_season_guid")]
    pub media_season_guid: Option<String>,
    #[serde(rename = "media_season_order")]
    pub media_season_order: Option<f64>, // decimal(20,4)
    #[serde(rename = "media_season_identifier")]
    pub media_season_identifier: Option<String>,
    #[serde(rename = "media_season_title")]
    pub media_season_title: Option<String>,
    #[serde(rename = "media_season_tags")]
    pub media_season_tags: Option<String>, // Comma-separated
    #[serde(rename = "media_series_guid")]
    pub media_series_guid: Option<String>,
    #[serde(rename = "media_series_original_audio_locale")]
    pub media_series_original_audio_locale: Option<String>,
    #[serde(rename = "media_series_original_audio_language")]
    pub media_series_original_audio_language: Option<String>,
    #[serde(rename = "media_publisher_name")]
    pub media_publisher_name: Option<String>,
    #[serde(rename = "media_series_title")]
    pub media_series_title: Option<String>,
    #[serde(rename = "media_series_tags")]
    pub media_series_tags: Option<String>, // Comma-separated
    #[serde(rename = "media_series_audio_locales")]
    pub media_series_audio_locales: Option<String>, // Comma-separated
    #[serde(rename = "media_series_audio_languages")]
    pub media_series_audio_languages: Option<String>, // Comma-separated
    #[serde(rename = "container_guid")]
    pub container_guid: Option<String>,
    #[serde(rename = "container_id")]
    pub container_id: Option<i64>, // bigint
    #[serde(rename = "container_type")]
    pub container_type: Option<String>,
    #[serde(rename = "container_order")]
    pub container_order: Option<f64>, // decimal(20,4)
    #[serde(rename = "container_title")]
    pub container_title: Option<String>,
    #[serde(rename = "container_description")]
    pub container_description: Option<String>,
    #[serde(rename = "container_genres")]
    pub container_genres: Option<String>, // Comma-separated
    #[serde(rename = "container_tags")]
    pub container_tags: Option<String>, // Comma-separated
    #[serde(rename = "container_audio_locale")]
    pub container_audio_locale: Option<String>,
    #[serde(rename = "container_audio_language")]
    pub container_audio_language: Option<String>,
    #[serde(rename = "container_channel_id")]
    pub container_channel_id: Option<i64>, // bigint
    #[serde(rename = "container_channel_name")]
    pub container_channel_name: Option<String>,
    #[serde(rename = "container_is_public")]
    pub container_is_public: Option<bool>,
    #[serde(rename = "container_is_clip")]
    pub container_is_clip: Option<bool>,
    #[serde(rename = "container_last_public_ts")]
    pub container_last_public_ts: Option<String>, // timestamp
    #[serde(rename = "container_is_subbed")]
    pub container_is_subbed: Option<bool>,
    #[serde(rename = "container_is_dubbed")]
    pub container_is_dubbed: Option<bool>,
    #[serde(rename = "container_available_offline")]
    pub container_available_offline: Option<bool>,
    #[serde(rename = "container_closed_captions_available")]
    pub container_closed_captions_available: Option<bool>,
    #[serde(rename = "container_episode_number")]
    pub container_episode_number: Option<String>,
    #[serde(rename = "container_episode_air_time")]
    pub container_episode_air_time: Option<String>, // timestamp
    #[serde(rename = "container_maturity_rating_id")]
    pub container_maturity_rating_id: Option<i64>, // bigint
    #[serde(rename = "container_maturity_advisory_scheme")]
    pub container_maturity_advisory_scheme: Option<String>,
    #[serde(rename = "container_maturity_advisory_code")]
    pub container_maturity_advisory_code: Option<String>,
    #[serde(rename = "container_maturity_level")]
    pub container_maturity_level: Option<i32>, // int
    #[serde(rename = "container_localized_items")]
    pub container_localized_items: Option<Vec<ContainerLocalizedItem>>,
    #[serde(rename = "container_season_guid")]
    pub container_season_guid: Option<String>,
    #[serde(rename = "container_season_order")]
    pub container_season_order: Option<f64>, // decimal(20,4)
    #[serde(rename = "container_season_title")]
    pub container_season_title: Option<String>,
    #[serde(rename = "container_season_tags")]
    pub container_season_tags: Option<String>, // Comma-separated
    #[serde(rename = "container_series_content_provider")]
    pub container_series_content_provider: Option<String>,
    #[serde(rename = "container_series_title")]
    pub container_series_title: Option<String>,
    #[serde(rename = "container_series_genres")]
    pub container_series_genres: Option<String>, // Comma-separated
    #[serde(rename = "container_series_tags")]
    pub container_series_tags: Option<String>, // Comma-separated
    #[serde(rename = "container_series_is_simulcast")]
    pub container_series_is_simulcast: Option<bool>,
    #[serde(rename = "container_series_launch_year")]
    pub container_series_launch_year: Option<i64>, // bigint
    #[serde(rename = "catalog_media_guid")]
    pub catalog_media_guid: Option<String>,
    #[serde(rename = "catalog_media_id")]
    pub catalog_media_id: Option<i64>, // bigint
    #[serde(rename = "catalog_media_type")]
    pub catalog_media_type: Option<String>,
    #[serde(rename = "catalog_media_title")]
    pub catalog_media_title: Option<String>,
    #[serde(rename = "catalog_media_genres")]
    pub catalog_media_genres: Option<String>, // Comma-separated
    #[serde(rename = "catalog_media_tags")]
    pub catalog_media_tags: Option<String>, // Comma-separated
    #[serde(rename = "catalog_media_duration_seconds")]
    pub catalog_media_duration_seconds: Option<f64>, // decimal(24,3)
    #[serde(rename = "catalog_media_audio_locale")]
    pub catalog_media_audio_locale: Option<String>,
    #[serde(rename = "catalog_media_audio_language")]
    pub catalog_media_audio_language: Option<String>,
    #[serde(rename = "catalog_media_hardsub_locale")]
    pub catalog_media_hardsub_locale: Option<String>,
    #[serde(rename = "catalog_media_hardsub_language")]
    pub catalog_media_hardsub_language: Option<String>,
    #[serde(rename = "catalog_media_localized_items")]
    pub catalog_media_localized_items: Option<Vec<CatalogLocalizedItem>>,
    #[serde(rename = "cr_media_guid")]
    pub cr_media_guid: Option<String>,
    #[serde(rename = "cr_media_type")]
    pub cr_media_type: Option<String>,
    #[serde(rename = "cr_media_title")]
    pub cr_media_title: Option<String>,
    #[serde(rename = "cr_media_tags")]
    pub cr_media_tags: Option<String>, // Comma-separated
    #[serde(rename = "cr_media_season_guid")]
    pub cr_media_season_guid: Option<String>,
    #[serde(rename = "cr_media_season_title")]
    pub cr_media_season_title: Option<String>,
    #[serde(rename = "cr_media_season_tags")]
    pub cr_media_season_tags: Option<String>, // Comma-separated
    #[serde(rename = "cr_media_series_guid")]
    pub cr_media_series_guid: Option<String>,
    #[serde(rename = "cr_media_series_title")]
    pub cr_media_series_title: Option<String>,
    #[serde(rename = "cr_media_series_original_audio_language")]
    pub cr_media_series_original_audio_language: Option<String>,
    #[serde(rename = "cr_media_publisher_name")]
    pub cr_media_publisher_name: Option<String>,
    #[serde(rename = "cr_media_series_tags")]
    pub cr_media_series_tags: Option<String>, // Comma-separated
    #[serde(rename = "airtable_primary_genre")]
    pub airtable_primary_genre: Option<String>,
    #[serde(rename = "airtable_secondary_genres")]
    pub airtable_secondary_genres: Option<String>, // Comma-separated
    #[serde(rename = "airtable_japanese_audience")]
    pub airtable_japanese_audience: Option<String>,
    #[serde(rename = "airtable_ca_brand_grade")]
    pub airtable_ca_brand_grade: Option<String>,
    #[serde(rename = "airtable_rating_descriptors_from_sp")]
    pub airtable_rating_descriptors_from_sp: Option<String>,
    #[serde(rename = "airtable_brand_grade_from_ca_data")]
    pub airtable_brand_grade_from_ca_data: Option<String>,
    #[serde(rename = "airtable_cr_rating_from_sp")]
    pub airtable_cr_rating_from_sp: Option<String>,
    #[serde(rename = "airtable_original_release_year")]
    pub airtable_original_release_year: Option<String>, // String, not int
    #[serde(rename = "airtable_content_tags")]
    pub airtable_content_tags: Option<String>, // Comma-separated
    #[serde(rename = "media_updated_ts")]
    pub media_updated_ts: Option<String>, // timestamp
                                          // ... other fields omitted for brevity ...
}

#[derive(Debug, Deserialize)]
pub struct ContainerLocalizedItem {
    pub event_timestamp: Option<String>, // timestamp
    pub id: Option<i64>,                 // bigint
    pub locale_code: Option<String>,
    pub name: Option<String>,
    pub value: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CatalogLocalizedItem {
    pub event_timestamp: Option<String>, // timestamp
    pub id: Option<i64>,                 // bigint
    pub locale_code: Option<String>,
    pub name: Option<String>,
    pub value: Option<String>,
}

// --- Engagement Struct ---

/// Represents a row in the `Engagement` table.
/// This is a denormalized table including view, user, and subscription info.
#[derive(Debug, Deserialize)]
pub struct EngagementRecord {
    #[serde(rename = "view_ts")]
    pub view_ts: Option<String>, // timestamp
    #[serde(rename = "view_date")]
    pub view_date: Option<String>, // date
    #[serde(rename = "anonymous_id")]
    pub anonymous_id: Option<String>,
    #[serde(rename = "view_media_id")]
    pub view_media_id: Option<String>,
    #[serde(rename = "view_subsidiary")]
    pub view_subsidiary: Option<String>,
    #[serde(rename = "view_etp_account_id")]
    pub view_etp_account_id: Option<String>,
    #[serde(rename = "view_context_app_platform")]
    pub view_context_app_platform: Option<String>,
    #[serde(rename = "view_context_device_type")]
    pub view_context_device_type: Option<String>,
    #[serde(rename = "view_context_os_name")]
    pub view_context_os_name: Option<String>,
    #[serde(rename = "view_subscription_plan")]
    pub view_subscription_plan: Option<String>,
    #[serde(rename = "view_profile_id")]
    pub view_profile_id: Option<String>,
    #[serde(rename = "view_seconds_watched")]
    pub view_seconds_watched: Option<f64>,
    #[serde(rename = "view_country_code_view")]
    pub view_country_code_view: Option<String>,
    #[serde(rename = "subscription_plan")]
    pub subscription_plan: Option<String>,
    #[serde(rename = "subscription_status")]
    pub subscription_status: Option<String>,
    #[serde(rename = "subscription_tenure_days")]
    pub subscription_tenure_days: Option<i64>, // bigint
    #[serde(rename = "account_country_code_account")]
    pub account_country_code_account: Option<String>,
    #[serde(rename = "account_crunchyroll_account_id")]
    pub account_crunchyroll_account_id: Option<i64>, // bigint
    #[serde(rename = "account_tenure_days")]
    pub account_tenure_days: Option<i32>, // int
    #[serde(rename = "region_country_name_account")]
    pub region_country_name_account: Option<String>,
    #[serde(rename = "region_major_account")]
    pub region_major_account: Option<String>,
    #[serde(rename = "region_minor_account")]
    pub region_minor_account: Option<String>,
    #[serde(rename = "catalog_media_genres")]
    pub catalog_media_genres: Option<String>,
    #[serde(rename = "catalog_media_is_dubbed")]
    pub catalog_media_is_dubbed: Option<bool>,
    #[serde(rename = "catalog_media_is_subbed")]
    pub catalog_media_is_subbed: Option<bool>,
    #[serde(rename = "catalog_show_id")]
    pub catalog_show_id: Option<String>,
    #[serde(rename = "catalog_show_tags")]
    pub catalog_show_tags: Option<String>,
    #[serde(rename = "catalog_show_title")]
    pub catalog_show_title: Option<String>,
    #[serde(rename = "catalog_show_genres")]
    pub catalog_show_genres: Option<String>,
    #[serde(rename = "catalog_original_release_year")]
    pub catalog_original_release_year: Option<String>,
    #[serde(rename = "catalog_show_primary_genre")]
    pub catalog_show_primary_genre: Option<String>,
    #[serde(rename = "catalog_show_secondary_genres")]
    pub catalog_show_secondary_genres: Option<String>,
    // ... other fields omitted for brevity ...
}

// --- Account Structs ---

/// Represents a row in the `Accounts` table.
#[derive(Debug, Deserialize)]
pub struct AccountRecord {
    #[serde(rename = "account_id")]
    pub account_id: Option<String>, // ETP account id
    #[serde(rename = "account_snapshot_ts")]
    pub account_snapshot_ts: Option<String>, // timestamp
    #[serde(rename = "account_profiles")]
    pub account_profiles: Option<AccountProfile>, // struct
    #[serde(rename = "account_profile_id")]
    pub account_profile_id: Option<String>,
    #[serde(rename = "account_profile_maturity_rating")]
    pub account_profile_maturity_rating: Option<String>,
    #[serde(rename = "account_profile_cf_preferred_app_language")]
    pub account_profile_cf_preferred_app_language: Option<String>,
    #[serde(rename = "account_profile_cf_preferred_content_audio_language")]
    pub account_profile_cf_preferred_content_audio_language: Option<String>,
    #[serde(rename = "account_profile_cf_preferred_content_subtitle_language")]
    pub account_profile_cf_preferred_content_subtitle_language: Option<String>,
    #[serde(rename = "account_created_ts")]
    pub account_created_ts: Option<String>, // timestamp
    #[serde(rename = "account_country_code")]
    pub account_country_code: Option<String>,
    #[serde(rename = "account_tenure_days")]
    pub account_tenure_days: Option<i32>, // int
    #[serde(rename = "account_external_id")]
    pub account_external_id: Option<i64>, // bigint
    #[serde(rename = "region_country_name")]
    pub region_country_name: Option<String>,
    #[serde(rename = "region_major")]
    pub region_major: Option<String>,
    #[serde(rename = "region_minor")]
    pub region_minor: Option<String>,
    // ... other fields omitted for brevity ...
}

#[derive(Debug, Deserialize)]
pub struct AccountProfile {
    pub profile_id: Option<String>,
    pub profile_name: Option<String>,
    pub maturity_rating: Option<String>,
    pub extended_maturity_rating: Option<String>,
    pub deleted: Option<i32>,
    pub avatar: Option<String>,
    pub profile_type: Option<String>,
    pub custom_fields: Option<HashMap<String, String>>,
    pub created: Option<String>,         // timestamp
    pub event_timestamp: Option<String>, // timestamp
}

// --- Subscription Structs ---

/// Represents a row in the `Subscriptions` table.
#[derive(Debug, Deserialize)]
pub struct SubscriptionRecord {
    #[serde(rename = "subscription_id")]
    pub subscription_id: Option<i64>, // bigint
    #[serde(rename = "subscription_processor")]
    pub subscription_processor: Option<String>,
    #[serde(rename = "subscription_snapshot_ts")]
    pub subscription_snapshot_ts: Option<String>, // timestamp
    #[serde(rename = "subscriber_id")]
    pub subscriber_id: Option<String>, // ETP account id
    #[serde(rename = "subscription_status")]
    pub subscription_status: Option<String>,
    #[serde(rename = "subscription_status_previous")]
    pub subscription_status_previous: Option<String>,
    #[serde(rename = "subscription_status_change_type")]
    pub subscription_status_change_type: Option<String>,
    #[serde(rename = "subscription_tenure_days")]
    pub subscription_tenure_days: Option<i64>, // bigint
    #[serde(rename = "subscription_etp_account_id")]
    pub subscription_etp_account_id: Option<String>,
    #[serde(rename = "subscription_country_code")]
    pub subscription_country_code: Option<String>,
    #[serde(rename = "subscription_first_created_ts")]
    pub subscription_first_created_ts: Option<String>, // timestamp
    #[serde(rename = "subscription_product_product_name")]
    pub subscription_product_product_name: Option<String>,
    #[serde(rename = "subscription_product_product_name_previous")]
    pub subscription_product_product_name_previous: Option<String>,
    #[serde(rename = "subscription_product_product_name_change_type")]
    pub subscription_product_product_name_change_type: Option<String>,
    #[serde(rename = "subscription_product_amount")]
    pub subscription_product_amount: Option<f64>,
    #[serde(rename = "subscription_product_currency_code")]
    pub subscription_product_currency_code: Option<String>,
    #[serde(rename = "subscription_product_free_trial_start_ts")]
    pub subscription_product_free_trial_start_ts: Option<String>, // timestamp
    #[serde(rename = "subscription_product_free_trial_end_ts")]
    pub subscription_product_free_trial_end_ts: Option<String>, // timestamp
    #[serde(rename = "subscription_product_conversion_ts")]
    pub subscription_product_conversion_ts: Option<String>, // timestamp
    #[serde(rename = "subscription_product_cancellation_requested_ts")]
    pub subscription_product_cancellation_requested_ts: Option<String>,
    #[serde(rename = "subscription_product_cancelled_ts")]
    pub subscription_product_cancelled_ts: Option<String>, // timestamp
    #[serde(rename = "subscription_product_days_to_convert")]
    pub subscription_product_days_to_convert: Option<i64>, // bigint
    #[serde(rename = "subscription_promo_coupon_code")]
    pub subscription_promo_coupon_code: Option<String>,
    #[serde(rename = "account_amazon_account_id")]
    pub account_amazon_account_id: Option<String>,
    #[serde(rename = "account_funimation_venue_id")]
    pub account_funimation_venue_id: Option<i32>, // int
    #[serde(rename = "account_country_code")]
    pub account_country_code: Option<String>,
    #[serde(rename = "account_registered_ts")]
    pub account_registered_ts: Option<String>, // timestamp
    #[serde(rename = "account_tenure_days")]
    pub account_tenure_days: Option<i64>, // bigint
    #[serde(rename = "region_country_name")]
    pub region_country_name: Option<String>,
    #[serde(rename = "region_major")]
    pub region_major: Option<String>,
    #[serde(rename = "region_minor")]
    pub region_minor: Option<String>,
    // ... other fields omitted for brevity ...
}
