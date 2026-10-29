import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mundusx_planner_service.media import media_plan


class MediaTests(unittest.TestCase):
    def test_client_image_delivery_tracks_client_execution_without_workspace_path(self):
        for model in ("Qwen/Qwen-Image", "Lykon/dreamshaper-8"):
            result = media_plan({"kind": "image", "image_model": model,
                "client_delivery": {"destination": "workspace", "tool_names": ["run_in_terminal"]}})
            delivery = result["client_delivery"]
            self.assertEqual(delivery["action"], "download_to_workspace")
            self.assertEqual(delivery["execution_owner"], "client")
            self.assertFalse(delivery["file_saved"])
            self.assertEqual(delivery["tool_names"], ["run_in_terminal"])
            self.assertEqual(result["profile_id"].startswith("dreamshaper"), model.startswith("Lykon"))
        self.assertNotIn("client_delivery", media_plan({"kind": "image"}))

    def test_client_delivery_rejects_unsupported_or_unbounded_requirements(self):
        for context in ({"destination": "server", "tool_names": ["terminal"]},
                        {"destination": "workspace", "tool_names": []},
                        {"destination": "workspace", "tool_names": ["bad name"]}, "workspace"):
            with self.assertRaises(ValueError):
                media_plan({"kind": "image", "client_delivery": context})

    def test_image_routes_to_async_queue_without_vision_llm_requirement(self):
        result = media_plan({"kind": "image"})
        self.assertEqual(result["execution_route"], "media_queue")
        self.assertEqual(result["operation"], "text_to_image")
        self.assertTrue(result["asynchronous"])
        self.assertEqual(result["node_selection_owner"], "media_scheduler")

    def test_t2v_and_i2v_choose_exact_verified_duration_profile(self):
        for seconds in range(1, 11):
            for reference in (False, True):
                result = media_plan({"kind": "video", "duration_seconds": seconds,
                                     "has_reference": reference})
                self.assertIn(f"-{seconds * 16 + 1}f-", result["profile_id"])
                self.assertEqual(result["operation"], "image_to_video" if reference else "text_to_video")

    def test_invalid_media_does_not_plan_llm_fallback(self):
        for item in ({"kind": "chat"}, {"kind": "image", "has_reference": True},
                     {"kind": "video", "duration_seconds": 11},
                     {"kind": "video", "duration_seconds": True},
                     {"kind": "image", "duration_seconds": 3}):
            with self.assertRaises(ValueError):
                media_plan(item)

    def test_fast_default_and_quality_modes_match_reference_operation(self):
        for reference in (False, True):
            fast=media_plan({"kind":"video","has_reference":reference})
            quality=media_plan({"kind":"video","has_reference":reference,"quality":"quality"})
            self.assertIn("lightning",fast["profile_id"])
            self.assertNotIn("lightning",quality["profile_id"])
            self.assertIn("480p",quality["profile_id"])
