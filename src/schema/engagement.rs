//! This file defines the Rust structs that correspond to your data schema.
//! These are primarily for type safety, documentation, and potential
//! row-based deserialization.

use serde::Deserialize;
use std::collections::HashMap;

// --- Nested Structs for Content Metadata ---

#[derive(Debug, Deserialize)]
pub struct LocalizedItem {
    pub event_timestamp: Option<String>, // timestamp
    pub id: Option<i64>,                 // bigint
    pub locale_code: Option<String>,
    pub name: Option<String>,
    pub value: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct GenresStruct {
    pub en: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct KeywordsStruct {
    pub en: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct SubtitleWindow {
    pub geo: Option<Vec<String>>,
    pub locale: Option<String>,
    pub watch_end: Option<String>,   // timestamp
    pub watch_start: Option<String>, // timestamp
}

#[derive(Debug, Deserialize)]
pub struct ContainerAdditionalAttributes {
    pub audio_locale: Option<String>,
    pub available_offline: Option<bool>,
    pub bg_color: Option<String>,
    pub closed_captions_available: Option<bool>,
    pub content_provider: Option<String>,
    pub editorial_rating: Option<i64>, // bigint
    pub episode: Option<String>,
    pub episode_air_date: Option<String>, // timestamp
    pub episode_production_id: Option<String>,
    pub genres: Option<GenresStruct>,
    pub is_auto_ingest: Option<bool>,
    pub is_dubbed: Option<bool>,
    pub is_master: Option<bool>,
    pub is_movie_auto_publish: Option<bool>,
    pub is_simulcast: Option<bool>,
    pub is_subbed: Option<bool>,
    pub keywords: Option<KeywordsStruct>,
    pub linked_guid: Option<String>,
    pub movie_release_year: Option<i64>, // bigint
    pub original_audio_locale: Option<String>,
    pub qc_failure_reason: Option<Vec<String>>,
    pub qc_notes: Option<String>,
    pub season_display_number: Option<String>,
    pub season_identifier: Option<String>,
    pub season_sequence_number: Option<i64>, // bigint
    pub season_tags: Option<HashMap<String, Vec<String>>>, // e.g., "en": [...]
    pub seo_description: Option<String>,
    pub seo_title: Option<String>,
    pub sequence_number: Option<i64>, // bigint
    pub series_launch_year: Option<i64>, // bigint
    pub subscriptions: Option<String>, // Could be more complex, String is safe
    pub subtitle_window: Option<Vec<SubtitleWindow>>,
    pub title: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ContainerRestrictionWindow {
    pub audio_locale: Option<String>,
    pub geo: Option<Vec<String>>,
    pub level: Option<Vec<String>>,
    pub list_end: Option<String>,   // timestamp
    pub list_start: Option<String>, // timestamp
    pub watch_end: Option<String>,  // timestamp
    pub watch_start: Option<String>, // timestamp
}

#[derive(Debug, Deserialize)]
pub struct ContainerSubtitleWindow {
    pub event_timestamp: Option<String>, // timestamp
    pub id: Option<i64>,                 // bigint
    pub locale: Option<String>,
    pub watch_start: Option<String>, // timestamp
    pub watch_end: Option<String>,   // timestamp
    pub geo_str: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct CatalogMediaAdditionalAttributes {
    pub ad_breaks: Option<Vec<f64>>,
    pub audio_locale: Option<String>,
    pub hardsub_locale: Option<String>,
    pub hd_flag: Option<bool>,
    pub mezzanine: Option<String>,
    pub sv2_hash: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CrMediaVideo {
    pub event_timestamp: Option<String>, // timestamp
    pub id: Option<i32>,                 // int
    pub itemhash: Option<String>,
    pub filename: Option<String>,
    #[serde(rename = "type")]
    pub type_field: Option<i32>, // int
    pub source: Option<String>,
    pub width: Option<i32>,  // int
    pub height: Option<i32>, // int
    pub framerate: Option<f64>, // decimal(20,4)
    #[serde(rename = "int")]
    pub int_field: Option<String>,
    pub aint: Option<String>,
    pub audiorate: Option<String>,
    pub duration: Option<f64>, // decimal(20,4)
    pub vcodec: Option<String>,
    pub acodec: Option<String>,
    pub container: Option<String>,
    pub created: Option<String>, // timestamp
    pub deleted: Option<i32>,    // int
    pub modified: Option<String>, // timestamp
    pub flags: Option<i32>,      // int
    pub encode_version: Option<String>,
    pub encode_info: Option<String>,
    pub cache_version: Option<i32>, // int
    pub format: Option<i32>,        // int
    pub format_primary: Option<i32>, // int
    pub hardsub_lang: Option<String>,
    pub audio_lang: Option<String>,
    pub version: Option<i32>, // int
    pub subtitle_update_time: Option<String>, // timestamp
}

#[derive(Debug, Deserialize)]
pub struct CrMediaVideoEncode {
    pub event_timestamp: Option<String>, // timestamp
    pub id: Option<i32>,                 // int
    pub itemhash: Option<String>,
    pub filename: Option<String>,
    pub video_id: Option<i32>, // int
    pub quality: Option<i32>,  // int
    pub source: Option<String>,
    pub width: Option<i32>,  // int
    pub height: Option<i32>, // int
    pub framerate: Option<f64>, // decimal(20,4)
    pub bitrate: Option<i32>,   // int
    pub abitrate: Option<i32>,  // int
    pub audiorate: Option<String>,
    pub duration: Option<f64>, // decimal(20,4)
    pub vcodec: Option<String>,
    pub acodec: Option<String>,
    pub container: Option<String>,
    pub created: Option<String>, // timestamp
    pub deleted: Option<i32>,    // int
    pub modified: Option<String>, // timestamp
    pub flags: Option<i32>,      // int
    pub encode_version: Option<String>,
    pub encode_info: Option<String>,
    pub deprecated: Option<i32>, // int
    pub old_video_id: Option<i32>, // int
}

// --- Main Content Metadata Struct ---

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
    #[serde(rename = "container_additional_attributes")]
    pub container_additional_attributes: Option<ContainerAdditionalAttributes>,
    #[serde(rename = "container_restriction_windows")]
    pub container_restriction_windows: Option<Vec<ContainerRestrictionWindow>>,
    #[serde(rename = "container_subtitle_windows")]
    pub container_subtitle_windows: Option<Vec<ContainerSubtitleWindow>>,
    #[serde(rename = "container_localized_items")]
    pub container_localized_items: Option<Vec<LocalizedItem>>, // Re-using LocalizedItem
    #[serde(rename = "container_season_guid")]
    pub container_season_guid: Option<String>,
    #[serde(rename = "container_parent_container_id")]
    pub container_parent_container_id: Option<i64>, // bigint
    #[serde(rename = "container_season_order")]
    pub container_season_order: Option<f64>, // decimal(20,4)
    #[serde(rename = "container_season_audio_locale")]
    pub container_season_audio_locale: Option<String>,
    #[serde(rename = "container_season_audio_language")]
    pub container_season_audio_language: Option<String>,
    #[serde(rename = "container_season_title")]
    pub container_season_title: Option<String>,
    #[serde(rename = "container_season_description")]
    pub container_season_description: Option<String>,
    #[serde(rename = "container_season_tags")]
    pub container_season_tags: Option<String>, // Comma-separated
    #[serde(rename = "container_season_display_number")]
    pub container_season_display_number: Option<String>,
    #[serde(rename = "container_season_identifier")]
    pub container_season_identifier: Option<String>,
    #[serde(rename = "container_season_sequence_number")]
    pub container_season_sequence_number: Option<i64>, // bigint
    #[serde(rename = "container_season_additional_attributes")]
    pub container_season_additional_attributes: Option<ContainerAdditionalAttributes>, // Re-using
    #[serde(rename = "container_parent_container_id_2")]
    pub container_parent_container_id_2: Option<i64>, // bigint
    #[serde(rename = "container_series_audio_locale")]
    pub container_series_audio_locale: Option<String>,
    #[serde(rename = "container_series_audio_language")]
    pub container_series_audio_language: Option<String>,
    #[serde(rename = "container_series_content_provider")]
    pub container_series_content_provider: Option<String>,
    #[serde(rename = "container_series_title")]
    pub container_series_title: Option<String>,
    #[serde(rename = "container_series_description")]
    pub container_series_description: Option<String>,
    #[serde(rename = "container_series_genres")]
    pub container_series_genres: Option<String>, // Comma-separated
    #[serde(rename = "container_series_tags")]
    pub container_series_tags: Option<String>, // Comma-separated
    #[serde(rename = "container_series_is_simulcast")]
    pub container_series_is_simulcast: Option<bool>,
    #[serde(rename = "container_series_launch_year")]
    pub container_series_launch_year: Option<i64>, // bigint
    #[serde(rename = "container_series_additional_attributes")]
    pub container_series_additional_attributes: Option<ContainerAdditionalAttributes>, // Re-using
    #[serde(rename = "catalog_media_guid")]
    pub catalog_media_guid: Option<String>,
    #[serde(rename = "catalog_media_id")]
    pub catalog_media_id: Option<i64>, // bigint
    #[serde(rename = "catalog_media_type")]
    pub catalog_media_type: Option<String>,
    #[serde(rename = "catalog_media_title")]
    pub catalog_media_title: Option<String>,
    #[serde(rename = "catalog_media_description")]
    pub catalog_media_description: Option<String>,
    #[serde(rename = "catalog_media_genres")]
    pub catalog_media_genres: Option<String>, // Comma-separated
    #[serde(rename = "catalog_media_tags")]
    pub catalog_media_tags: Option<String>, // Comma-separated
    #[serde(rename = "catalog_media_duration_seconds")]
    pub catalog_media_duration_seconds: Option<f64>, // decimal(24,3)
    #[serde(rename = "catalog_media_runtime")]
    pub catalog_media_runtime: Option<String>,
    #[serde(rename = "catalog_media_audio_locale")]
    pub catalog_media_audio_locale: Option<String>,
    #[serde(rename = "catalog_media_audio_language")]
    pub catalog_media_audio_language: Option<String>,
    #[serde(rename = "catalog_media_hardsub_locale")]
    pub catalog_media_hardsub_locale: Option<String>,
    #[serde(rename = "catalog_media_hardsub_language")]
    pub catalog_media_hardsub_language: Option<String>,
    #[serde(rename = "catalog_media_sv2_hash")]
    pub catalog_media_sv2_hash: Option<String>,
    #[serde(rename = "catalog_media_ad_breaks")]
    pub catalog_media_ad_breaks: Option<String>, // e.g., "[89,715,1300]"
    #[serde(rename = "catalog_media_hd_flag")]
    pub catalog_media_hd_flag: Option<String>, // "TRUE"
    #[serde(rename = "catalog_media_additional_attributes")]
    pub catalog_media_additional_attributes: Option<CatalogMediaAdditionalAttributes>,
    #[serde(rename = "catalog_media_localized_items")]
    pub catalog_media_localized_items: Option<Vec<LocalizedItem>>, // Re-using LocalizedItem
    #[serde(rename = "cr_media_guid")]
    pub cr_media_guid: Option<String>,
    #[serde(rename = "cr_media_id")]
    pub cr_media_id: Option<i32>, // int
    #[serde(rename = "cr_media_type")]
    pub cr_media_type: Option<String>,
    #[serde(rename = "cr_media_duration_seconds")]
    pub cr_media_duration_seconds: Option<f64>,
    #[serde(rename = "cr_media_runtime")]
    pub cr_media_runtime: Option<String>,
    #[serde(rename = "cr_media_order")]
    pub cr_media_order: Option<f64>, // decimal(20,4)
    #[serde(rename = "cr_media_audio_locale")]
    pub cr_media_audio_locale: Option<String>,
    #[serde(rename = "cr_media_audio_language")]
    pub cr_media_audio_language: Option<String>,
    #[serde(rename = "cr_media_hardsub_locale")]
    pub cr_media_hardsub_locale: Option<String>,
    #[serde(rename = "cr_media_hardsub_language")]
    pub cr_media_hardsub_language: Option<String>,
    #[serde(rename = "cr_media_title")]
    pub cr_media_title: Option<String>,
    #[serde(rename = "cr_media_description")]
    pub cr_media_description: Option<String>,
    #[serde(rename = "cr_media_air_time")]
    pub cr_media_air_time: Option<String>, // timestamp
    #[serde(rename = "cr_media_available_time")]
    pub cr_media_available_time: Option<String>, // timestamp
    #[serde(rename = "cr_media_unavailable_time")]
    pub cr_media_unavailable_time: Option<String>, // timestamp
    #[serde(rename = "cr_media_premium_available_time")]
    pub cr_media_premium_available_time: Option<String>, // timestamp
    #[serde(rename = "cr_media_premium_unavailable_time")]
    pub cr_media_premium_unavailable_time: Option<String>, // timestamp
    #[serde(rename = "cr_media_generally_available_time")]
    pub cr_media_generally_available_time: Option<String>, // timestamp
    #[serde(rename = "cr_media_generally_unavailable_time")]
    pub cr_media_generally_unavailable_time: Option<String>, // timestamp
    #[serde(rename = "cr_media_tags")]
    pub cr_media_tags: Option<String>, // Comma-separated
    #[serde(rename = "cr_media_videos")]
    pub cr_media_videos: Option<Vec<CrMediaVideo>>,
    #[serde(rename = "cr_media_video_encodes")]
    pub cr_media_video_encodes: Option<Vec<CrMediaVideoEncode>>,
    #[serde(rename = "cr_media_season_guid")]
    pub cr_media_season_guid: Option<String>,
    #[serde(rename = "cr_media_season_id")]
    pub cr_media_season_id: Option<i32>, // int
    #[serde(rename = "cr_media_season_title")]
    pub cr_media_season_title: Option<String>,
    #[serde(rename = "cr_media_season_order")]
    pub cr_media_season_order: Option<i32>, // int
    #[serde(rename = "cr_media_season_identifier")]
    pub cr_media_season_identifier: Option<String>,
    #[serde(rename = "cr_media_season_tags")]
    pub cr_media_season_tags: Option<String>, // Comma-separated
    #[serde(rename = "cr_media_series_guid")]
    pub cr_media_series_guid: Option<String>,
    #[serde(rename = "cr_media_series_id")]
    pub cr_media_series_id: Option<i32>, // int
    #[serde(rename = "cr_media_series_title")]
    pub cr_media_series_title: Option<String>,
    #[serde(rename = "cr_media_series_slug")]
    pub cr_media_series_slug: Option<String>,
    #[serde(rename = "cr_media_series_original_audio_locale")]
    pub cr_media_series_original_audio_locale: Option<String>,
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
    #[serde(rename = "airtable_age_rating_from_sp")]
    pub airtable_age_rating_from_sp: Option<String>,
    #[serde(rename = "airtable_brand_grade_from_ca_data")]
    pub airtable_brand_grade_from_ca_data: Option<String>,
    #[serde(rename = "airtable_cr_rating_from_sp")]
    pub airtable_cr_rating_from_sp: Option<String>,
    #[serde(rename = "airtable_original_release_year")]
    pub airtable_original_release_year: Option<String>,
    #[serde(rename = "airtable_content_tags")]
    pub airtable_content_tags: Option<String>, // Comma-separated
    #[serde(rename = "airtable_original_production_studios")]
    pub airtable_original_production_studios: Option<String>,
    #[serde(rename = "airtable_min_original_release_date_airtable")]
    pub airtable_min_original_release_date_airtable: Option<String>,
    #[serde(rename = "airtable_max_original_release_date_airtable")]
    pub airtable_max_original_release_date_airtable: Option<String>,
    #[serde(rename = "airtable_extended_maturity_rating")]
    pub airtable_extended_maturity_rating: Option<String>,
    #[serde(rename = "media_updated_ts")]
    pub media_updated_ts: Option<String>, // timestamp
}

// --- Engagement Struct ---

/// Represents a row in the `Engagement` table.
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
    #[serde(rename = "amazon_rl_media_id")]
    pub amazon_rl_media_id: Option<String>,
    #[serde(rename = "view_subsidiary")]
    pub view_subsidiary: Option<String>,
    #[serde(rename = "view_etp_account_id")]
    pub view_etp_account_id: Option<String>,
    #[serde(rename = "view_amazon_account_id")]
    pub view_amazon_account_id: Option<String>,
    #[serde(rename = "view_context_app_platform")]
    pub view_context_app_platform: Option<String>,
    #[serde(rename = "view_context_device_type")]
    pub view_context_device_type: Option<String>,
    #[serde(rename = "view_context_os_name")]
    pub view_context_os_name: Option<String>,
    #[serde(rename = "view_context_os_version")]
    pub view_context_os_version: Option<String>,
    #[serde(rename = "view_context_device_manufacturer")]
    pub view_context_device_manufacturer: Option<String>,
    #[serde(rename = "view_context_device_model")]
    pub view_context_device_model: Option<String>,
    #[serde(rename = "view_context_device_name")]
    pub view_context_device_name: Option<String>,
    #[serde(rename = "view_context_app_build")]
    pub view_context_app_build: Option<String>,
    #[serde(rename = "view_context_app_name")]
    pub view_context_app_name: Option<String>,
    #[serde(rename = "view_context_app_version")]
    pub view_context_app_version: Option<String>,
    #[serde(rename = "view_start_ts")]
    pub view_start_ts: Option<String>, // timestamp
    #[serde(rename = "view_end_ts")]
    pub view_end_ts: Option<String>, // timestamp
    #[serde(rename = "view_media_subtitle_language")]
    pub view_media_subtitle_language: Option<String>,
    #[serde(rename = "view_player_sdk")]
    pub view_player_sdk: Option<String>,
    #[serde(rename = "view_subscription_plan")]
    pub view_subscription_plan: Option<String>,
    #[serde(rename = "view_profile_id")]
    pub view_profile_id: Option<String>,
    #[serde(rename = "view_playback_source")]
    pub view_playback_source: Option<String>,
    #[serde(rename = "view_playback_type")]
    pub view_playback_type: Option<String>,
    #[serde(rename = "view_closed_caption_language")]
    pub view_closed_caption_language: Option<String>,
    #[serde(rename = "view_seconds_watched")]
    pub view_seconds_watched: Option<f64>, // double
    #[serde(rename = "view_country_code_view")]
    pub view_country_code_view: Option<String>,
    #[serde(rename = "view_platform_name")]
    pub view_platform_name: Option<String>,
    #[serde(rename = "view_context_network_wifi")]
    pub view_context_network_wifi: Option<bool>,
    #[serde(rename = "view_context_network_cellular")]
    pub view_context_network_cellular: Option<bool>,
    #[serde(rename = "viewership_attribution_source_screen")]
    pub viewership_attribution_source_screen: Option<String>,
    #[serde(rename = "viewership_attribution_source_collection")]
    pub viewership_attribution_source_collection: Option<String>,
    #[serde(rename = "music_media_anime_ids")]
    pub music_media_anime_ids: Option<String>,
    #[serde(rename = "music_media_anime_titles")]
    pub music_media_anime_titles: Option<String>,
    #[serde(rename = "music_media_artist_id")]
    pub music_media_artist_id: Option<String>,
    #[serde(rename = "music_media_artist_name")]
    pub music_media_artist_name: Option<String>,
    #[serde(rename = "music_media_artist_objects")]
    pub music_media_artist_objects: Option<String>,
    #[serde(rename = "music_media_slug")]
    pub music_media_slug: Option<String>,
    #[serde(rename = "subscription_plan")]
    pub subscription_plan: Option<String>,
    #[serde(rename = "subscription_status")]
    pub subscription_status: Option<String>,
    #[serde(rename = "subscription_status_change_type")]
    pub subscription_status_change_type: Option<String>,
    #[serde(rename = "subscription_status_previous")]
    pub subscription_status_previous: Option<String>,
    #[serde(rename = "subscription_tenure_days")]
    pub subscription_tenure_days: Option<i64>, // bigint
    #[serde(rename = "account_country_code_account")]
    pub account_country_code_account: Option<String>,
    #[serde(rename = "account_crunchyroll_account_id")]
    pub account_crunchyroll_account_id: Option<i64>, // bigint
    #[serde(rename = "account_funimation_migration_ts")]
    pub account_funimation_migration_ts: Option<String>, // timestamp
    #[serde(rename = "account_funimation_venue_id")]
    pub account_funimation_venue_id: Option<i32>, // int
    #[serde(rename = "account_tenure_days")]
    pub account_tenure_days: Option<i32>, // int
    #[serde(rename = "account_profile_type")]
    pub account_profile_type: Option<String>,
    #[serde(rename = "region_country_name_account")]
    pub region_country_name_account: Option<String>,
    #[serde(rename = "region_major_account")]
    pub region_major_account: Option<String>,
    #[serde(rename = "region_minor_account")]
    pub region_minor_account: Option<String>,
    #[serde(rename = "region_portal_account")]
    pub region_portal_account: Option<String>,
    #[serde(rename = "region_country_name_view")]
    pub region_country_name_view: Option<String>,
    #[serde(rename = "region_major_view")]
    pub region_major_view: Option<String>,
    #[serde(rename = "region_minor_view")]
    pub region_minor_view: Option<String>,
    #[serde(rename = "catalog_media_episode_air_ts")]
    pub catalog_media_episode_air_ts: Option<String>, // timestamp
    #[serde(rename = "catalog_media_audio_language")]
    pub catalog_media_audio_language: Option<String>,
    #[serde(rename = "catalog_media_audio_locale")]
    pub catalog_media_audio_locale: Option<String>,
    #[serde(rename = "catalog_media_channel_id")]
    pub catalog_media_channel_id: Option<i64>, // bigint
    #[serde(rename = "catalog_media_channel_name")]
    pub catalog_media_channel_name: Option<String>,
    #[serde(rename = "catalog_media_duration")]
    pub catalog_media_duration: Option<f64>, // double
    #[serde(rename = "catalog_media_episode_number")]
    pub catalog_media_episode_number: Option<String>,
    #[serde(rename = "catalog_media_generally_available_ts")]
    pub catalog_media_generally_available_ts: Option<String>, // timestamp
    #[serde(rename = "catalog_media_genres")]
    pub catalog_media_genres: Option<String>, // Comma-separated
    #[serde(rename = "catalog_media_is_dubbed")]
    pub catalog_media_is_dubbed: Option<bool>,
    #[serde(rename = "catalog_media_is_subbed")]
    pub catalog_media_is_subbed: Option<bool>,
    #[serde(rename = "catalog_media_licensor_name")]
    pub catalog_media_licensor_name: Option<String>,
    #[serde(rename = "catalog_media_maturity_rating_id")]
    pub catalog_media_maturity_rating_id: Option<i64>, // bigint
    #[serde(rename = "catalog_media_migrated_from_funi")]
    pub catalog_media_migrated_from_funi: Option<bool>,
    #[serde(rename = "catalog_media_premium_available_ts")]
    pub catalog_media_premium_available_ts: Option<String>, // timestamp
    #[serde(rename = "catalog_media_runtime")]
    pub catalog_media_runtime: Option<String>,
    #[serde(rename = "catalog_season_number")]
    pub catalog_season_number: Option<f64>, // decimal(20,4)
    #[serde(rename = "catalog_season_tags")]
    pub catalog_season_tags: Option<String>, // Comma-separated
    #[serde(rename = "catalog_season_title")]
    pub catalog_season_title: Option<String>,
    #[serde(rename = "catalog_show_id")]
    pub catalog_show_id: Option<String>,
    #[serde(rename = "catalog_media_show_languages")]
    pub catalog_media_show_languages: Option<String>, // Comma-separated
    #[serde(rename = "catalog_show_original_audio_locale")]
    pub catalog_show_original_audio_locale: Option<String>,
    #[serde(rename = "catalog_show_tags")]
    pub catalog_show_tags: Option<String>, // Comma-separated
    #[serde(rename = "catalog_show_title")]
    pub catalog_show_title: Option<String>,
    #[serde(rename = "catalog_media_tags")]
    pub catalog_media_tags: Option<String>, // Comma-separated
    #[serde(rename = "catalog_media_title")]
    pub catalog_media_title: Option<String>,
    #[serde(rename = "catalog_show_genres")]
    pub catalog_show_genres: Option<String>, // Comma-separated
    #[serde(rename = "catalog_original_release_year")]
    pub catalog_original_release_year: Option<String>,
    #[serde(rename = "catalog_show_primary_genre")]
    pub catalog_show_primary_genre: Option<String>,
    #[serde(rename = "catalog_show_secondary_genres")]
    pub catalog_show_secondary_genres: Option<String>, // Comma-separated
    #[serde(rename = "viewership_access_level")]
    pub viewership_access_level: Option<String>,
    #[serde(rename = "process_ts")]
    pub process_ts: Option<String>, // timestamp
}