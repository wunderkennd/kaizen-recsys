//! This file defines the Rust structs that correspond to your data schemas.
//! These are primarily for type safety, documentation, and potential
//! row-based deserialization.

use serde::Deserialize;
use std::collections::HashMap;

// --- Nested Structs for Account ---

#[derive(Debug, Deserialize)]
pub struct AccountProfile {
    pub profile_id: Option<String>,
    pub profile_name: Option<String>,
    pub maturity_rating: Option<String>,
    pub extended_maturity_rating: Option<String>,
    pub deleted: Option<i32>, // int
    pub avatar: Option<String>,
    pub profile_type: Option<String>,
    pub custom_fields: Option<HashMap<String, String>>,
    pub created: Option<String>,         // timestamp
    pub event_timestamp: Option<String>, // timestamp
}

#[derive(Debug, Deserialize)]
pub struct IdentityProvider {
    pub identity_provider: Option<String>,
    pub identity_provider_user_id: Option<String>,
    pub region: Option<String>,
}

// --- Main Account Struct ---

/// Represents a row in the `Accounts` table.
#[derive(Debug, Deserialize)]
pub struct AccountRecord {
    #[serde(rename = "account_id")]
    pub account_id: Option<String>,
    #[serde(rename = "account_tenant")]
    pub account_tenant: Option<String>,
    #[serde(rename = "account_snapshot_ts")]
    pub account_snapshot_ts: Option<String>, // timestamp
    #[serde(rename = "account_profiles")]
    pub account_profiles: Option<AccountProfile>, // struct
    #[serde(rename = "account_profile_id")]
    pub account_profile_id: Option<String>,
    #[serde(rename = "account_profile_name")]
    pub account_profile_name: Option<String>,
    #[serde(rename = "account_profile_maturity_rating")]
    pub account_profile_maturity_rating: Option<String>,
    #[serde(rename = "account_profile_extended_maturity_rating")]
    pub account_profile_extended_maturity_rating: Option<HashMap<String, String>>, // map<string,string>
    #[serde(rename = "account_profile_deleted")]
    pub account_profile_deleted: Option<bool>,
    #[serde(rename = "account_profile_avatar")]
    pub account_profile_avatar: Option<String>,
    #[serde(rename = "account_profile_type")]
    pub account_profile_type: Option<String>,
    #[serde(rename = "account_profile_cf_age_consent")]
    pub account_profile_cf_age_consent: Option<bool>,
    #[serde(rename = "account_profile_cf_autoplay")]
    pub account_profile_cf_autoplay: Option<bool>,
    #[serde(rename = "account_profile_cf_country")]
    pub account_profile_cf_country: Option<String>,
    #[serde(rename = "account_profile_cf_cr_beta_opt_in")]
    pub account_profile_cf_cr_beta_opt_in: Option<bool>,
    #[serde(rename = "account_profile_cf_crleg_email_verified")]
    pub account_profile_cf_crleg_email_verified: Option<bool>,
    #[serde(rename = "account_profile_cf_do_not_sell")]
    pub account_profile_cf_do_not_sell: Option<bool>,
    #[serde(rename = "account_profile_cf_exclude_from_reporting")]
    pub account_profile_cf_exclude_from_reporting: Option<bool>,
    #[serde(rename = "account_profile_cf_extended_maturity_rating")]
    pub account_profile_cf_extended_maturity_rating: Option<HashMap<String, String>>, // map<string,string>
    #[serde(rename = "account_profile_cf_mature_content_flag_manga")]
    pub account_profile_cf_mature_content_flag_manga: Option<bool>,
    #[serde(rename = "account_profile_cf_performance_test_user")]
    pub account_profile_cf_performance_test_user: Option<bool>,
    #[serde(rename = "account_profile_cf_preferred_app_language")]
    pub account_profile_cf_preferred_app_language: Option<String>,
    #[serde(rename = "account_profile_cf_preferred_communication_language")]
    pub account_profile_cf_preferred_communication_language: Option<String>,
    #[serde(rename = "account_profile_cf_preferred_content_audio_language")]
    pub account_profile_cf_preferred_content_audio_language: Option<String>,
    #[serde(rename = "account_profile_cf_preferred_content_subtitle_language")]
    pub account_profile_cf_preferred_content_subtitle_language: Option<String>,
    #[serde(rename = "account_profile_cf_preferred_default_video_quality")]
    pub account_profile_cf_preferred_default_video_quality: Option<String>,
    #[serde(rename = "account_profile_cf_public_profile_enabled")]
    pub account_profile_cf_public_profile_enabled: Option<bool>,
    #[serde(rename = "account_profile_cf_qa_user")]
    pub account_profile_cf_qa_user: Option<bool>,
    #[serde(rename = "account_profile_cf_redacted")]
    pub account_profile_cf_redacted: Option<bool>,
    #[serde(rename = "account_profile_cf_sound")]
    pub account_profile_cf_sound: Option<bool>,
    #[serde(rename = "account_profile_cf_wallpaper")]
    pub account_profile_cf_wallpaper: Option<String>,
    #[serde(rename = "account_credentials_email")]
    pub account_credentials_email: Option<String>,
    #[serde(rename = "account_credentials_phone")]
    pub account_credentials_phone: Option<String>,
    #[serde(rename = "account_credentials_username")]
    pub account_credentials_username: Option<String>,
    #[serde(rename = "account_credentials_password")]
    pub account_credentials_password: Option<String>,
    #[serde(rename = "account_identities_no_new_sessions_allowed")]
    pub account_identities_no_new_sessions_allowed: Option<bool>,
    #[serde(rename = "account_identities_email_verified")]
    pub account_identities_email_verified: Option<bool>,
    #[serde(rename = "account_identities_force_password_reset")]
    pub account_identities_force_password_reset: Option<bool>,
    #[serde(rename = "account_identities_status")]
    pub account_identities_status: Option<String>,
    #[serde(rename = "account_identities_disabled")]
    pub account_identities_disabled: Option<bool>,
    #[serde(rename = "account_identities_contact_cs")]
    pub account_identities_contact_cs: Option<bool>,
    #[serde(rename = "account_identity_providers")]
    pub account_identity_providers: Option<Vec<IdentityProvider>>, // array<struct<...>>
    #[serde(rename = "account_amazon_identity_provider_user_id")]
    pub account_amazon_identity_provider_user_id: Option<String>,
    #[serde(rename = "account_amazon_identity_provider_region")]
    pub account_amazon_identity_provider_region: Option<String>,
    #[serde(rename = "account_subscriptions_subscription_id")]
    pub account_subscriptions_subscription_id: Option<i64>, // bigint
    #[serde(rename = "account_subscriptions_country_code")]
    pub account_subscriptions_country_code: Option<String>,
    #[serde(rename = "account_created_ts")]
    pub account_created_ts: Option<String>, // timestamp
    #[serde(rename = "account_updated_ts")]
    pub account_updated_ts: Option<String>, // timestamp
    #[serde(rename = "account_profile_created_ts")]
    pub account_profile_created_ts: Option<String>, // timestamp
    #[serde(rename = "account_profile_updated_ts")]
    pub account_profile_updated_ts: Option<String>, // timestamp
    #[serde(rename = "account_login_history_country_code")]
    pub account_login_history_country_code: Option<String>,
    #[serde(rename = "account_country_code")]
    pub account_country_code: Option<String>,
    #[serde(rename = "account_tenure_days")]
    pub account_tenure_days: Option<i32>, // int
    #[serde(rename = "account_external_id")]
    pub account_external_id: Option<i64>, // bigint
    #[serde(rename = "account_profile_is_test")]
    pub account_profile_is_test: Option<bool>,
    #[serde(rename = "account_first_country_code")]
    pub account_first_country_code: Option<String>,
    #[serde(rename = "account_is_test")]
    pub account_is_test: Option<bool>,
    #[serde(rename = "region_country_name")]
    pub region_country_name: Option<String>,
    #[serde(rename = "region_major")]
    pub region_major: Option<String>,
    #[serde(rename = "region_minor")]
    pub region_minor: Option<String>,
    #[serde(rename = "region_portal")]
    pub region_portal: Option<String>,
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
}
