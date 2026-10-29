import unittest
from mundusx_planner_service.artifact import artifact_plan


class ArtifactTests(unittest.TestCase):
    def test_formats_require_real_encoders_and_unknown_formats_fail_closed(self):
        for fmt in ("png", "jpeg", "webp", "bmp", "tiff", "gif", "avif"):
            plan = artifact_plan({"format": fmt, "prompt": "Create a still image of a cat"})
            self.assertEqual(plan["action"], "generate")
            self.assertEqual(plan["operation"], "encode_image")
        for fmt in ("svg", "xyz", "exe", "pdf"):
            self.assertEqual(artifact_plan({"format": fmt, "prompt": "create cat"})["action"], "unsupported")

    def test_scratch_is_a_project_not_an_image_extension(self):
        plan = artifact_plan({"format": "sb3", "prompt": "create walking cat with hat"})
        self.assertEqual(plan["operation"], "scratch_sprite")
        self.assertEqual(plan["motion"], "walk")
        self.assertFalse(plan["file_saved"])
        self.assertEqual(artifact_plan({"format": "sb3", "prompt": "Create a multiplayer game with scores"})["action"], "unsupported")

    def test_animations_do_not_become_still_images(self):
        self.assertEqual(artifact_plan({"format": "gif", "prompt": "create animated cat"})["action"], "unsupported")
