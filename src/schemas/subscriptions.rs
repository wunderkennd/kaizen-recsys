//! This file defines the Rust structs that correspond to your data schemas.
//! These are primarily for type safety, documentation, and potential
//! row-based deserialization.

use serde::Deserialize;
use std::collections::HashMap;

// --- Nested Structs for Subscription ---

#[derive(Debug, Deserialize)]
pub struct SubscriptionProductTag {
    pub tag_name: Option<String>,
    pub tag_value: Option<String>,
}

// --- Main Subscription Struct ---

/// Represents a row in the `Subscriptions` table.
#[derive(Debug, Deserialize)]
pub struct SubscriptionRecord {
    #[serde(rename = "subscription_id")]
    pub subscription_id: Option<i64>, // bigint
    #[serde(rename = "subscription_processor_id")]
    pub subscription_processor_id: Option<i32>, // int
    #[serde(rename = "subscription_processor")]
    pub subscription_processor: Option<String>,
    #[serde(rename = "subscription_sub_processor")]
    pub subscription_sub_processor: Option<String>,
    #[serde(rename = "subscription_snapshot_ts")]
    pub subscription_snapshot_ts: Option<String>, // timestamp
    #[serde(rename = "subscription_snapshot_date")]
    pub subscription_snapshot_date: Option<String>, // date
    #[serde(rename = "subscription_subsidiary")]
    pub subscription_subsidiary: Option<String>,
    #[serde(rename = "subscriber_id")]
    pub subscriber_id: Option<String>,
    #[serde(rename = "subscription_status")]
    pub subscription_status: Option<String>,
    #[serde(rename = "subscription_status_previous")]
    pub subscription_status_previous: Option<String>,
    #[serde(rename = "subscription_status_change_type")]
    pub subscription_status_change_type: Option<String>,
    #[serde(rename = "subscription_streak")]
    pub subscription_streak: Option<i32>, // int
    #[serde(rename = "subscription_tenure_days")]
    pub subscription_tenure_days: Option<i64>, // bigint
    #[serde(rename = "subscription_processor_streak")]
    pub subscription_processor_streak: Option<i32>, // int
    #[serde(rename = "subscription_processor_tenure_days")]
    pub subscription_processor_tenure_days: Option<i64>, // bigint
    #[serde(rename = "subscription_product_name_streak")]
    pub subscription_product_name_streak: Option<i32>, // int
    #[serde(rename = "subscription_product_name_tenure_days")]
    pub subscription_product_name_tenure_days: Option<i64>, // bigint
    #[serde(rename = "subscription_cycle_start_ts")]
    pub subscription_cycle_start_ts: Option<String>, // timestamp
    #[serde(rename = "subscription_cycle_tenure_days")]
    pub subscription_cycle_tenure_days: Option<i64>, // bigint
    #[serde(rename = "subscription_cycle_end_ts")]
    pub subscription_cycle_end_ts: Option<String>, // timestamp
    #[serde(rename = "subscription_billing_cycle_start_ts")]
    pub subscription_billing_cycle_start_ts: Option<String>, // timestamp
    #[serde(rename = "subscription_billing_cycle_end_ts")]
    pub subscription_billing_cycle_end_ts: Option<String>, // timestamp
    #[serde(rename = "subscription_cr_account_id")]
    pub subscription_cr_account_id: Option<i64>, // bigint
    #[serde(rename = "subscription_etp_account_id")]
    pub subscription_etp_account_id: Option<String>,
    #[serde(rename = "subscription_country_code")]
    pub subscription_country_code: Option<String>,
    #[serde(rename = "subscription_first_created_ts")]
    pub subscription_first_created_ts: Option<String>, // timestamp
    #[serde(rename = "subscription_product_product_id")]
    pub subscription_product_product_id: Option<i64>, // bigint
    #[serde(rename = "subscription_product_product_type")]
    pub subscription_product_product_type: Option<String>,
    #[serde(rename = "subscription_product_product_sku")]
    pub subscription_product_product_sku: Option<String>,
    #[serde(rename = "subscription_product_product_name")]
    pub subscription_product_product_name: Option<String>,
    #[serde(rename = "subscription_product_product_name_previous")]
    pub subscription_product_product_name_previous: Option<String>,
    #[serde(rename = "subscription_product_product_name_change_type")]
    pub subscription_product_product_name_change_type: Option<String>,
    #[serde(rename = "subscription_product_tier_change_type")]
    pub subscription_product_tier_change_type: Option<String>,
    #[serde(rename = "subscription_product_product_description")]
    pub subscription_product_product_description: Option<String>,
    #[serde(rename = "subscription_product_product_cycle_duration")]
    pub subscription_product_product_cycle_duration: Option<String>,
    #[serde(rename = "subscription_product_id")]
    pub subscription_product_id: Option<i64>, // bigint
    #[serde(rename = "subscription_product_event_ts")]
    pub subscription_product_event_ts: Option<String>, // timestamp
    #[serde(rename = "subscription_product_source_reference")]
    pub subscription_product_source_reference: Option<String>,
    #[serde(rename = "subscription_product_amount")]
    pub subscription_product_amount: Option<f64>, // double
    #[serde(rename = "subscription_product_currency_code")]
    pub subscription_product_currency_code: Option<String>,
    #[serde(rename = "subscription_product_free_trial_start_ts")]
    pub subscription_product_free_trial_start_ts: Option<String>, // timestamp
    #[serde(rename = "subscription_product_free_trial_end_ts")]
    pub subscription_product_free_trial_end_ts: Option<String>, // timestamp
    #[serde(rename = "subscription_product_conversion_ts")]
    pub subscription_product_conversion_ts: Option<String>, // timestamp
    #[serde(rename = "subscription_product_cancellation_requested_ts")]
    pub subscription_product_cancellation_requested_ts: Option<String>, // timestamp
    #[serde(rename = "subscription_product_cancelled_ts")]
    pub subscription_product_cancelled_ts: Option<String>, // timestamp
    #[serde(rename = "subscription_product_cancellation_type")]
    pub subscription_product_cancellation_type: Option<String>,
    #[serde(rename = "subscription_product_days_to_convert")]
    pub subscription_product_days_to_convert: Option<i64>, // bigint
    #[serde(rename = "subscription_product_tags")]
    pub subscription_product_tags: Option<Vec<SubscriptionProductTag>>, // array<struct<...>>
    #[serde(rename = "subscription_product_referrer")]
    pub subscription_product_referrer: Option<String>,
    #[serde(rename = "subscription_product_expiration_reason_code")]
    pub subscription_product_expiration_reason_code: Option<String>,
    #[serde(rename = "subscription_product_expiration_reason_category")]
    pub subscription_product_expiration_reason_category: Option<String>,
    #[serde(rename = "subscription_product_expiration_reason_subcategory")]
    pub subscription_product_expiration_reason_subcategory: Option<String>,
    #[serde(rename = "subscription_promo_coupon_code")]
    pub subscription_promo_coupon_code: Option<String>,
    #[serde(rename = "subscription_promo_campaign_id")]
    pub subscription_promo_campaign_id: Option<String>,
    #[serde(rename = "subscription_promo_campaign_name")]
    pub subscription_promo_campaign_name: Option<String>,
    #[serde(rename = "subscription_promo_source")]
    pub subscription_promo_source: Option<String>,
    #[serde(rename = "subscription_promo_country_code")]
    pub subscription_promo_country_code: Option<String>,
    #[serde(rename = "subscription_promo_format")]
    pub subscription_promo_format: Option<String>,
    #[serde(rename = "subscription_promo_retailer")]
    pub subscription_promo_retailer: Option<String>,
    #[serde(rename = "subscription_info_is_projected")]
    pub subscription_info_is_projected: Option<bool>,
    #[serde(rename = "subscription_is_latest_snapshot")]
    pub subscription_is_latest_snapshot: Option<bool>,
    #[serde(rename = "viewership_pre_signup_first_view_media_id")]
    pub viewership_pre_signup_first_view_media_id: Option<String>,
    #[serde(rename = "viewership_pre_signup_first_view_media_title")]
    pub viewership_pre_signup_first_view_media_title: Option<String>,
    #[serde(rename = "viewership_pre_signup_first_view_media_season_title")]
    pub viewership_pre_signup_first_view_media_season_title: Option<String>,
    #[serde(rename = "viewership_pre_signup_first_view_media_series_title")]
    pub viewership_pre_signup_first_view_media_series_title: Option<String>,
    #[serde(rename = "viewership_pre_signup_first_view_media_air_time")]
    pub viewership_pre_signup_first_view_media_air_time: Option<String>, // timestamp
    #[serde(rename = "viewership_pre_signup_first_view_cr_media_premium_available_time")]
    pub viewership_pre_signup_first_view_cr_media_premium_available_time: Option<String>, // timestamp
    #[serde(rename = "viewership_pre_signup_first_view_media_series_original_audio_language")]
    pub viewership_pre_signup_first_view_media_series_original_audio_language: Option<String>,
    #[serde(rename = "viewership_pre_signup_first_view_container_season_audio_language")]
    pub viewership_pre_signup_first_view_container_season_audio_language: Option<String>,
    #[serde(rename = "viewership_pre_signup_first_view_ts")]
    pub viewership_pre_signup_first_view_ts: Option<String>, // timestamp
    #[serde(rename = "viewership_pre_signup_last_view_media_id")]
    pub viewership_pre_signup_last_view_media_id: Option<String>,
    #[serde(rename = "viewership_pre_signup_last_view_media_title")]
    pub viewership_pre_signup_last_view_media_title: Option<String>,
    #[serde(rename = "viewership_pre_signup_last_view_media_season_title")]
    pub viewership_pre_signup_last_view_media_season_title: Option<String>,
    #[serde(rename = "viewership_pre_signup_last_view_media_series_title")]
    pub viewership_pre_signup_last_view_media_series_title: Option<String>,
    #[serde(rename = "viewership_pre_signup_last_view_media_air_time")]
    pub viewership_pre_signup_last_view_media_air_time: Option<String>, // timestamp
    #[serde(rename = "viewership_pre_signup_last_view_cr_media_premium_available_time")]
    pub viewership_pre_signup_last_view_cr_media_premium_available_time: Option<String>, // timestamp
    #[serde(rename = "viewership_pre_signup_last_view_media_series_original_audio_language")]
    pub viewership_pre_signup_last_view_media_series_original_audio_language: Option<String>,
    #[serde(rename = "viewership_pre_signup_last_view_container_season_audio_language")]
    pub viewership_pre_signup_last_view_container_season_audio_language: Option<String>,
    #[serde(rename = "viewership_pre_signup_last_view_ts")]
    pub viewership_pre_signup_last_view_ts: Option<String>, // timestamp
    #[serde(rename = "viewership_post_signup_first_view_media_id")]
    pub viewership_post_signup_first_view_media_id: Option<String>,
    #[serde(rename = "viewership_post_signup_first_view_media_title")]
    pub viewership_post_signup_first_view_media_title: Option<String>,
    #[serde(rename = "viewership_post_signup_first_view_media_season_title")]
    pub viewership_post_signup_first_view_media_season_title: Option<String>,
    #[serde(rename = "viewership_post_signup_first_view_media_series_title")]
    pub viewership_post_signup_first_view_media_series_title: Option<String>,
    #[serde(rename = "viewership_post_signup_first_view_media_air_time")]
    pub viewership_post_signup_first_view_media_air_time: Option<String>, // timestamp
    #[serde(rename = "viewership_post_signup_first_view_cr_media_premium_available_time")]
    pub viewership_post_signup_first_view_cr_media_premium_available_time: Option<String>, // timestamp
    #[serde(rename = "viewership_post_signup_first_view_media_series_original_audio_language")]
    pub viewership_post_signup_first_view_media_series_original_audio_language: Option<String>,
    #[serde(rename = "viewership_post_signup_first_view_container_season_audio_language")]
    pub viewership_post_signup_first_view_container_season_audio_language: Option<String>,
    #[serde(rename = "viewership_post_signup_first_view_ts")]
    pub viewership_post_signup_first_view_ts: Option<String>, // timestamp
    #[serde(rename = "viewership_post_signup_last_view_media_id")]
    pub viewership_post_signup_last_view_media_id: Option<String>,
    #[serde(rename = "viewership_post_signup_last_view_media_title")]
    pub viewership_post_signup_last_view_media_title: Option<String>,
    #[serde(rename = "viewership_post_signup_last_view_media_season_title")]
    pub viewership_post_signup_last_view_media_season_title: Option<String>,
    #[serde(rename = "viewership_post_signup_last_view_media_series_title")]
    pub viewership_post_signup_last_view_media_series_title: Option<String>,
    #[serde(rename = "viewership_post_signup_last_view_media_air_time")]
    pub viewership_post_signup_last_view_media_air_time: Option<String>, // timestamp
    #[serde(rename = "viewership_post_signup_last_view_cr_media_premium_available_time")]
    pub viewership_post_signup_last_view_cr_media_premium_available_time: Option<String>, // timestamp
    #[serde(rename = "viewership_post_signup_last_view_media_series_original_audio_language")]
    pub viewership_post_signup_last_view_media_series_original_audio_language: Option<String>,
    #[serde(rename = "viewership_post_signup_last_view_container_season_audio_language")]
    pub viewership_post_signup_last_view_container_season_audio_language: Option<String>,
    #[serde(rename = "viewership_post_signup_last_view_ts")]
    pub viewership_post_signup_last_view_ts: Option<String>, // timestamp
    #[serde(rename = "viewership_last_view_media_id")]
    pub viewership_last_view_media_id: Option<String>,
    #[serde(rename = "viewership_last_view_media_title")]
    pub viewership_last_view_media_title: Option<String>,
    #[serde(rename = "viewership_last_view_media_season_title")]
    pub viewership_last_view_media_season_title: Option<String>,
    #[serde(rename = "viewership_last_view_media_series_title")]
    pub viewership_last_view_media_series_title: Option<String>,
    #[serde(rename = "viewership_last_view_media_air_time")]
    pub viewership_last_view_media_air_time: Option<String>, // timestamp
    #[serde(rename = "viewership_last_view_cr_media_premium_available_time")]
    pub viewership_last_view_cr_media_premium_available_time: Option<String>, // timestamp
    #[serde(rename = "viewership_last_view_media_series_original_audio_language")]
    pub viewership_last_view_media_series_original_audio_language: Option<String>,
    #[serde(rename = "viewership_last_view_container_season_audio_language")]
    pub viewership_last_view_container_season_audio_language: Option<String>,
    #[serde(rename = "viewership_last_view_ts")]
    pub viewership_last_view_ts: Option<String>, // timestamp
    #[serde(rename = "viewership_view_seconds_watched")]
    pub viewership_view_seconds_watched: Option<f64>, // double
    #[serde(rename = "store_pre_signup_first_order_id")]
    pub store_pre_signup_first_order_id: Option<String>,
    #[serde(rename = "store_pre_signup_first_order_ts")]
    pub store_pre_signup_first_order_ts: Option<String>, // timestamp
    #[serde(rename = "store_pre_signup_last_order_id")]
    pub store_pre_signup_last_order_id: Option<String>,
    #[serde(rename = "store_pre_signup_last_order_ts")]
    pub store_pre_signup_last_order_ts: Option<String>, // timestamp
    #[serde(rename = "store_post_signup_first_order_id")]
    pub store_post_signup_first_order_id: Option<String>,
    #[serde(rename = "store_post_signup_first_order_ts")]
    pub store_post_signup_first_order_ts: Option<String>, // timestamp
    #[serde(rename = "store_post_signup_last_order_id")]
    pub store_post_signup_last_order_id: Option<String>,
    #[serde(rename = "store_post_signup_last_order_ts")]
    pub store_post_signup_last_order_ts: Option<String>, // timestamp
    #[serde(rename = "store_last_order_id")]
    pub store_last_order_id: Option<String>,
    #[serde(rename = "store_last_order_ts")]
    pub store_last_order_ts: Option<String>, // timestamp
    #[serde(rename = "store_order_total")]
    pub store_order_total: Option<f64>, // double
    #[serde(rename = "game_pre_signup_first_game_id")]
    pub game_pre_signup_first_game_id: Option<String>,
    #[serde(rename = "game_pre_signup_first_game_name")]
    pub game_pre_signup_first_game_name: Option<String>,
    #[serde(rename = "game_pre_signup_first_game_device_type")]
    pub game_pre_signup_first_game_device_type: Option<String>,
    #[serde(rename = "game_pre_signup_first_game_ts")]
    pub game_pre_signup_first_game_ts: Option<String>, // timestamp
    #[serde(rename = "game_pre_signup_last_game_id")]
    pub game_pre_signup_last_game_id: Option<String>,
    #[serde(rename = "game_pre_signup_last_game_name")]
    pub game_pre_signup_last_game_name: Option<String>,
    #[serde(rename = "game_pre_signup_last_game_device_type")]
    pub game_pre_signup_last_game_device_type: Option<String>,
    #[serde(rename = "game_pre_signup_last_game_ts")]
    pub game_pre_signup_last_game_ts: Option<String>, // timestamp
    #[serde(rename = "game_post_signup_first_game_id")]
    pub game_post_signup_first_game_id: Option<String>,
    #[serde(rename = "game_post_signup_first_game_name")]
    pub game_post_signup_first_game_name: Option<String>,
    #[serde(rename = "game_post_signup_first_game_device_type")]
    pub game_post_signup_first_game_device_type: Option<String>,
    #[serde(rename = "game_post_signup_first_game_ts")]
    pub game_post_signup_first_game_ts: Option<String>, // timestamp
    #[serde(rename = "game_post_signup_last_game_id")]
    pub game_post_signup_last_game_id: Option<String>,
    #[serde(rename = "game_post_signup_last_game_name")]
    pub game_post_signup_last_game_name: Option<String>,
    #[serde(rename = "game_post_signup_last_game_device_type")]
    pub game_post_signup_last_game_device_type: Option<String>,
    #[serde(rename = "game_post_signup_last_game_ts")]
    pub game_post_signup_last_game_ts: Option<String>, // timestamp
    #[serde(rename = "game_last_game_id")]
    pub game_last_game_id: Option<String>,
    #[serde(rename = "game_last_game_name")]
    pub game_last_game_name: Option<String>,
    #[serde(rename = "game_last_game_device_type")]
    pub game_last_game_device_type: Option<String>,
    #[serde(rename = "game_last_game_ts")]
    pub game_last_game_ts: Option<String>, // timestamp
    #[serde(rename = "payment_method_processor")]
    pub payment_method_processor: Option<String>,
    #[serde(rename = "payment_method_service_provider")]
    pub payment_method_service_provider: Option<String>,
    #[serde(rename = "payment_invoice_id")]
    pub payment_invoice_id: Option<String>,
    #[serde(rename = "payment_invoice_subs_invoice_id")]
    pub payment_invoice_subs_invoice_id: Option<i64>, // bigint
    #[serde(rename = "payment_method_country_code")]
    pub payment_method_country_code: Option<String>,
    #[serde(rename = "payment_invoice_currency_code")]
    pub payment_invoice_currency_code: Option<String>,
    #[serde(rename = "payment_invoice_status")]
    pub payment_invoice_status: Option<String>,
    #[serde(rename = "payment_transaction_status")]
    pub payment_transaction_status: Option<String>,
    #[serde(rename = "payment_transaction_ts")]
    pub payment_transaction_ts: Option<String>, // timestamp
    #[serde(rename = "payment_last_success_transaction_ts")]
    pub payment_last_success_transaction_ts: Option<String>, // timestamp
    #[serde(rename = "payment_verified_method_processor")]
    pub payment_verified_method_processor: Option<String>,
    #[serde(rename = "payment_verified_method_service_provider")]
    pub payment_verified_method_service_provider: Option<String>,
    #[serde(rename = "payment_verified_billing_country")]
    pub payment_verified_billing_country: Option<String>,
    #[serde(rename = "payment_verified_checkout_type")]
    pub payment_verified_checkout_type: Option<String>,
    #[serde(rename = "payment_verified_zip_code")]
    pub payment_verified_zip_code: Option<String>,
    #[serde(rename = "payment_verified_transaction_ts")]
    pub payment_verified_transaction_ts: Option<String>, // timestamp
    #[serde(rename = "account_amazon_account_id")]
    pub account_amazon_account_id: Option<String>,
    #[serde(rename = "account_funimation_venue_id")]
    pub account_funimation_venue_id: Option<i32>, // int
    #[serde(rename = "account_funimation_migration_ts")]
    pub account_funimation_migration_ts: Option<String>, // timestamp
    #[serde(rename = "account_wakanim_id")]
    pub account_wakanim_id: Option<String>, // varchar(65535)
    #[serde(rename = "account_wakanim_migration_ts")]
    pub account_wakanim_migration_ts: Option<String>, // timestamp_ntz
    #[serde(rename = "account_vrv_id")]
    pub account_vrv_id: Option<i64>, // bigint
    #[serde(rename = "account_vrv_migration_ts")]
    pub account_vrv_migration_ts: Option<String>, // timestamp_ntz
    #[serde(rename = "account_country_code")]
    pub account_country_code: Option<String>,
    #[serde(rename = "account_registered_ts")]
    pub account_registered_ts: Option<String>, // timestamp
    #[serde(rename = "account_profiles")]
    pub account_profiles: Option<i64>, // bigint
    #[serde(rename = "account_case_closed_profiles")]
    pub account_case_closed_profiles: Option<i64>, // bigint
    #[serde(rename = "account_tenure_days")]
    pub account_tenure_days: Option<i64>, // bigint
    #[serde(rename = "region_country_name")]
    pub region_country_name: Option<String>,
    #[serde(rename = "region_major")]
    pub region_major: Option<String>,
    #[serde(rename = "region_minor")]
    pub region_minor: Option<String>,
}
