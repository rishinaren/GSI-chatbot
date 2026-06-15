import unittest

from standards_rag.video import (
    VideoTranscript,
    VideoTranscriptStore,
    extract_youtube_id,
    video_request_explicit,
)


def _sample_store() -> VideoTranscriptStore:
    store = VideoTranscriptStore()
    store.add(
        VideoTranscript(
            video_id="ds",
            youtube_id="abcdefghijk",
            title="ASTM D5321 Direct Shear of Geosynthetic Interfaces",
            transcript=(
                "The direct shear test determines the shear strength of geosynthetic "
                "interfaces under increasing normal stress."
            ),
            standards=("D5321",),
        )
    )
    store.add(
        VideoTranscript(
            video_id="perm",
            youtube_id="lmnopqrstuv",
            title="ASTM D4491 Water Permeability of Geotextiles by Permittivity",
            transcript="Permittivity measures the water flow rate through a geotextile.",
            standards=("D4491",),
        )
    )
    return store


class VideoHelpersTests(unittest.TestCase):
    def test_extract_youtube_id_from_url(self) -> None:
        self.assertEqual(extract_youtube_id("https://www.youtube.com/watch?v=abcdefghijk"), "abcdefghijk")
        self.assertEqual(extract_youtube_id("https://youtu.be/abcdefghijk"), "abcdefghijk")
        self.assertEqual(extract_youtube_id("abcdefghijk"), "abcdefghijk")
        self.assertIsNone(extract_youtube_id("short"))
        self.assertIsNone(extract_youtube_id(""))

    def test_video_request_explicit(self) -> None:
        self.assertTrue(video_request_explicit("show me a video about direct shear"))
        self.assertTrue(video_request_explicit("is there a youtube clip for this?"))
        self.assertFalse(video_request_explicit("what does ASTM D5321 cover?"))


class VideoSearchTests(unittest.TestCase):
    def test_matches_relevant_video(self) -> None:
        store = _sample_store()
        matches = store.search("direct shear strength of geosynthetic interfaces", top_k=2)
        self.assertTrue(matches)
        self.assertEqual(matches[0].video.video_id, "ds")

    def test_standard_designation_boost(self) -> None:
        store = _sample_store()
        matches = store.search("D4491 permittivity", top_k=1)
        self.assertTrue(matches)
        self.assertEqual(matches[0].video.video_id, "perm")

    def test_embed_url_format(self) -> None:
        store = _sample_store()
        match = store.search("direct shear", top_k=1)[0]
        self.assertEqual(match.video.embed_url, "https://www.youtube.com/embed/abcdefghijk")
        self.assertIn("embed_url", match.to_dict())


if __name__ == "__main__":
    unittest.main()
