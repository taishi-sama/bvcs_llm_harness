//! Bounds Bevy's global font atlas cache.
//!
//! Bevy caches rasterized glyphs globally in [`FontAtlasSet`]. The cache is not
//! tied to card entities: despawning transcript cards frees their layout data,
//! but every rasterized glyph stays in the atlas for the lifetime of the app.
//! Long conversations with lots of distinct characters (or fallback fonts) can
//! therefore keep growing VRAM even when only 20 cards are alive.
//!
//! Bevy does not expose per-glyph eviction, only a global clear. This module
//! watches the atlas size, clears it once it crosses a configurable limit, and
//! asks the text systems to re-rasterize the currently visible text.

use std::{
    env,
    time::{Duration, Instant},
};

use bevy::{
    asset::Assets,
    image::Image,
    log::{info, warn},
    prelude::*,
    text::{FontAtlasSet, FontHinting},
};

const DEFAULT_MAX_MB: u64 = 32;
const DEFAULT_CHECK_INTERVAL_MS: u64 = 250;
const DEFAULT_COOLDOWN_MS: u64 = 1_000;

/// Runtime limits for the shared font atlas cache.
#[derive(Resource)]
pub(crate) struct GlyphCacheLimit {
    /// Hard upper bound for atlas image bytes. `None` disables the limiter.
    max_bytes: Option<u64>,
    /// How often the atlas is measured.
    check_interval: Duration,
    /// Minimum time between clears, so the atlas cannot be cleared every frame
    /// when the visible working set itself is larger than the limit.
    cooldown: Duration,
    last_check: Option<Instant>,
    last_clear: Option<Instant>,
}

impl Default for GlyphCacheLimit {
    fn default() -> Self {
        let max_mb = env::var("FONT_ATLAS_MAX_MB")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_MAX_MB);
        let check_ms = env_duration("FONT_ATLAS_CHECK_MS", DEFAULT_CHECK_INTERVAL_MS);
        let cooldown_ms = env_duration("FONT_ATLAS_COOLDOWN_MS", DEFAULT_COOLDOWN_MS);

        Self {
            max_bytes: (max_mb > 0).then_some(max_mb * 1024 * 1024),
            check_interval: Duration::from_millis(check_ms),
            cooldown: Duration::from_millis(cooldown_ms),
            last_check: None,
            last_clear: None,
        }
    }
}

fn env_duration(name: &str, default_ms: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default_ms)
}

/// Clear the global glyph atlas when it grows beyond [`GlyphCacheLimit`].
///
/// The clear happens in `Update`, before Bevy's text systems run in
/// `PostUpdate`. Marking [`FontHinting`] changed forces both regular UI text and
/// editable text inputs to rebuild their glyph lists and re-add only the glyphs
/// that are actually needed by the current frames.
pub(crate) fn limit_glyph_cache(
    mut atlases: ResMut<FontAtlasSet>,
    images: Res<Assets<Image>>,
    mut hints: Query<&mut FontHinting>,
    mut limit: ResMut<GlyphCacheLimit>,
) {
    let Some(max_bytes) = limit.max_bytes else {
        return;
    };

    let now = Instant::now();
    if limit
        .last_check
        .is_some_and(|last| now.duration_since(last) < limit.check_interval)
    {
        return;
    }
    limit.last_check = Some(now);

    if limit
        .last_clear
        .is_some_and(|last| now.duration_since(last) < limit.cooldown)
    {
        return;
    }

    let total = atlases.total_bytes(&images);
    if total <= max_bytes {
        return;
    }

    atlases.clear();
    // Text systems only rebuild when their inputs change; the atlas is not part
    // of their change detection. Bumping the hinting component re-rasterizes all
    // visible text (including editable inputs) in the same frame.
    for mut hint in &mut hints {
        hint.set_changed();
    }
    limit.last_clear = Some(now);

    warn!(
        "cleared font glyph cache: {:.1} MiB exceeded the {:.1} MiB limit",
        total as f64 / (1024. * 1024.),
        max_bytes as f64 / (1024. * 1024.)
    );
}

/// Log the active limit once so VRAM issues are easier to diagnose.
pub(crate) fn log_glyph_cache_limit(limit: Res<GlyphCacheLimit>) {
    match limit.max_bytes {
        Some(max) => info!(
            "font glyph cache limit: {:.1} MiB",
            max as f64 / (1024. * 1024.)
        ),
        None => info!("font glyph cache limit disabled"),
    }
}
