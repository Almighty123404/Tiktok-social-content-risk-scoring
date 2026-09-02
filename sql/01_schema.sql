-- ============================================================
-- Trust & Safety Risk Analytics — Schema
-- Table: videos
-- Source: TikTok user engagement dataset (Kaggle)
-- ============================================================

DROP TABLE IF EXISTS videos;

CREATE TABLE videos (
    row_id                  INTEGER PRIMARY KEY,
    claim_status             TEXT,       -- 'claim' or 'opinion'
    video_id                 INTEGER,
    video_duration_sec       INTEGER,
    video_transcription_text TEXT,
    verified_status           TEXT,       -- 'verified' / 'not verified'
    author_ban_status         TEXT,       -- 'active' / 'under review' / 'banned'
    video_view_count          REAL,
    video_like_count          REAL,
    video_share_count         REAL,
    video_download_count      REAL,
    video_comment_count       REAL
);
