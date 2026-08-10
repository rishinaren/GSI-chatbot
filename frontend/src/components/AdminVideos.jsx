import { useMemo, useState } from "react";
import { listAdminVideos } from "../api";
import { PageHead, useAdminData } from "./AdminLayout";
import { SearchIcon, VideoIcon } from "./AdminIcons";

// A video earns its place in an answer by covering a standard the library
// holds. So the useful thing to show an administrator is not the video list -
// it is that link, and where it is missing: a designation tagged "not in the
// library" is a standard worth uploading, which is the one action this screen
// should prompt.

export default function AdminVideos({ head }) {
  const { data, loading, error } = useAdminData(listAdminVideos, "Could not load the videos.");
  const [query, setQuery] = useState("");
  const [onlyGaps, setOnlyGaps] = useState(false);

  const videos = data?.videos ?? [];
  const missing = useMemo(
    () => videos.filter((video) => video.standards.some((entry) => !entry.in_library)),
    [videos],
  );

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return videos.filter((video) => {
      if (onlyGaps && !video.standards.some((entry) => !entry.in_library)) return false;
      if (!needle) return true;
      return (
        video.title.toLowerCase().includes(needle) ||
        video.standards.some((entry) => entry.designation.toLowerCase().includes(needle))
      );
    });
  }, [videos, query, onlyGaps]);

  if (loading) {
    return (
      <>
        <PageHead head={head} />
        <p className="library-empty">Loading…</p>
      </>
    );
  }

  if (error) {
    return (
      <>
        <PageHead head={head} />
        <div className="library-error">{error}</div>
      </>
    );
  }

  return (
    <>
      <PageHead
        head={head}
        sub={`${data.linked_count} of ${data.video_count} videos are linked to a standard in the library.`}
      />

      <div className="library-toolbar">
        <div className="library-search">
          <SearchIcon />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search by title or standard number"
            aria-label="Search the videos"
          />
        </div>
        <div className="library-chips">
          <button
            type="button"
            className={`library-chip ${onlyGaps ? "" : "active"}`}
            onClick={() => setOnlyGaps(false)}
          >
            All
            <span className="library-chip-count">{videos.length}</span>
          </button>
          <button
            type="button"
            className={`library-chip ${onlyGaps ? "active" : ""}`}
            onClick={() => setOnlyGaps(true)}
          >
            Missing a standard
            <span className="library-chip-count">{missing.length}</span>
          </button>
        </div>
      </div>

      <div className="library-list">
        {visible.length === 0 ? (
          <p className="library-empty">
            {onlyGaps
              ? "Every standard named in a video title is in the library."
              : "No videos match what you typed."}
          </p>
        ) : null}

        {visible.map((video) => (
          <div key={video.video_id} className="admin-video-row">
            {/* Thumbnails come from YouTube, so they can be slow or blocked
                entirely. The icon sits underneath: a pending or failed image
                leaves a recognisable placeholder rather than an empty box. */}
            <a
              className="admin-video-thumb"
              href={video.youtube_url}
              target="_blank"
              rel="noopener noreferrer"
              aria-label={`Watch ${video.title} on YouTube`}
            >
              <span className="admin-video-thumb-icon" aria-hidden="true">
                <VideoIcon />
              </span>
              <img src={video.thumbnail_url} alt="" loading="lazy" />
            </a>
            <div className="admin-video-main">
              <a
                className="admin-video-title"
                href={video.youtube_url}
                target="_blank"
                rel="noopener noreferrer"
              >
                {video.title}
              </a>
              <div className="admin-video-tags">
                {video.standards.length === 0 ? (
                  <span className="library-pill">No standard in the title</span>
                ) : (
                  video.standards.map((entry) => (
                    <span
                      key={entry.designation}
                      className={`admin-tag ${entry.in_library ? "in" : "out"}`}
                      title={
                        entry.in_library
                          ? `${entry.designation} is in the library`
                          : `${entry.designation} is not in the library yet`
                      }
                    >
                      {entry.designation}
                      {entry.in_library ? "" : " · not in the library"}
                    </span>
                  ))
                )}
              </div>
            </div>
            <div className="library-row-meta">
              <span className={video.linked ? "admin-linked" : "admin-unlinked"}>
                {video.linked ? "Linked" : "Not linked"}
              </span>
              {video.has_transcript ? <span>Transcript read</span> : <span>No transcript</span>}
            </div>
          </div>
        ))}
      </div>
    </>
  );
}
