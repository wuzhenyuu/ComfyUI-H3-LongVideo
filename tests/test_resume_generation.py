"""Selective generation and durable queue recovery, with no GPU or network calls."""
import asyncio
import copy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
import uuid
import wave
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("h3lv_test")
package.__path__ = [str(ROOT)]
sys.modules.setdefault("h3lv_test", package)
core = importlib.import_module("h3lv_test.core")
controller = importlib.import_module("h3lv_test.controller")
REAL_PROBE = controller.probe_video
REAL_ASSEMBLE = controller.assemble


def snapshot(video="7", value="current"):
    return {
        "loader_id": "1", "video_id": video,
        "output_contract_version": controller.OUTPUT_CONTRACT_VERSION,
        "prompt_rule_source": "workflow",
        "prompt": {
            "1": {"class_type": "H3LVUnified", "inputs": {}},
            "2": {"class_type": "Sampler", "inputs": {"value": value}},
            video: {"class_type": "VHS_VideoCombine", "inputs": {}},
        },
    }


class Queue:
    def __init__(self, case):
        self.case = case
        self.indices, self.prompts, self.history, self.active = [], [], {}, set()
        self.on_put = None
        self.fail = False

    def put(self, item):
        index = item[2]["1"]["inputs"]["segment_index"]
        self.indices.append(index)
        self.prompts.append(copy.deepcopy(item[2]))
        if self.fail:
            self.history[item[1]] = {"status": {"status_str": "error"}, "outputs": {}}
        else:
            video_id = next(key for key, node in item[2].items()
                            if node["class_type"] == "VHS_VideoCombine")
            path = self.case.directory / f"new_{index}_{len(self.indices)}.mp4"
            path.write_bytes(b"generated-video")
            self.history[item[1]] = self.case.video_history(path, video_id)
        if self.on_put:
            self.on_put(item)

    def get_history(self, prompt_id):
        return {prompt_id: self.history[prompt_id]} if prompt_id in self.history else {}

    def get_current_queue(self):
        return [], [(0, item) for item in self.active]


class ResumeGenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.root = self.output / "projects"
        self.plan = {
            "id": uuid.uuid4().hex, "revision": 3, "approved": True,
            "run_status": "paused", "sample_rate": 48000, "samples": 192000,
            "duration": 4, "mode": "singing", "created": 0,
            "segments": [{"index": index, "start_sample": index * 48000,
                          "end_sample": (index + 1) * 48000, "prompt": f"take-{index}",
                          "generation_frames": 24, "edit_frames": 24} for index in range(4)],
        }
        self.plan["approved_fingerprint"] = core.fingerprint(self.plan)
        core.write_plan(self.root, self.plan)
        self.directory = core.project_path(self.root, self.plan["id"])
        self.global_snapshot = core.state_file(self.directory, "queue_snapshot.json")
        self.global_snapshot.write_text(json.dumps(snapshot(value="original")), encoding="utf-8")
        self.queue = Queue(self)
        self.server = types.SimpleNamespace(number=0, prompt_queue=self.queue)
        self.payload = {**snapshot(), "resume_unfinished": True, "revision": 3}
        async def validate(*args):
            return True, None, ["7"], {}
        modules = {
            "execution": types.SimpleNamespace(validate_prompt=validate),
            "folder_paths": types.SimpleNamespace(get_output_directory=lambda: str(self.output)),
        }
        self.enterContext(patch.dict(sys.modules, modules))
        self.probe = self.enterContext(patch.object(controller, "probe_video", return_value={
            "width": 160, "height": 96, "r_frame_rate": "24/1", "nb_read_frames": "24"}))
        self.assembly = self.enterContext(patch.object(controller, "assemble",
            return_value=str(self.output / "final.mp4")))
        self.enterContext(patch.object(controller, "record_error"))
        diagnostics = importlib.import_module("h3lv_test.diagnostics")
        self.enterContext(patch.object(diagnostics, "record_error"))
        self.addCleanup(controller.TASKS.pop, self.plan["id"], None)

    def save(self):
        core.write_plan(self.root, self.plan)

    def load(self):
        return core.read_plan(self.root, self.plan["id"])

    def completed(self, index, stale=False):
        path = self.directory / f"old_{index}.mp4"
        path.write_bytes(f"old-video-{index}".encode())
        row = self.plan["segments"][index]
        row["job"] = {"status": "completed", "video": str(path),
                      "prompt_id": f"old-{index}", "input_fingerprint": "old-input"}
        if stale:
            row.update(needs_regeneration=True, regeneration_reason="changed material")
        return path

    def video_history(self, path, video_id="7"):
        return {"status": {"status_str": "success"}, "outputs": {video_id: {"gifs": [{
            "type": "output", "filename": path.name,
            "subfolder": path.parent.relative_to(self.output).as_posix()}]}}}

    def frozen_files(self):
        return {path.relative_to(self.directory).as_posix():
                (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                for path in self.directory.rglob("*") if path.is_file()}

    def start_without_execute(self, payload=None):
        def capture(coroutine):
            coroutine.close()
            return MagicMock()
        with patch.object(controller.asyncio, "create_task", side_effect=capture):
            controller.start(self.root, self.plan["id"], payload or self.payload, self.server)
        controller.TASKS.pop(self.plan["id"], None)

    async def run_generation(self, payload=None):
        controller.start(self.root, self.plan["id"], payload or self.payload, self.server)
        task = controller.TASKS[self.plan["id"]]
        await task
        return self.load()

    async def test_mixed_resume_preserves_old_takes_and_snapshot_and_only_submits_unfinished(self):
        old = [self.completed(0), self.completed(1, stale=True)]
        self.plan["segments"][2]["job"] = {"status": "failed", "prompt_id": "failure", "error": "bad"}
        self.save()
        protected = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                     for path in [self.global_snapshot, *old]}
        before_rows = copy.deepcopy(self.plan["segments"][:2])
        result = await self.run_generation()
        self.assertEqual(self.queue.indices, [2, 3])
        self.assertEqual(result["segments"][:2], before_rows)
        self.assertEqual(result["segments"][2]["failed_attempts"][0]["prompt_id"], "failure")
        self.assertEqual(result["run_mode"], "resume_unfinished")
        self.assertTrue(result["final_stale"])
        self.assertEqual(result["final_retained_stale_segments"], [1])
        self.assertEqual(result["run_status"], "completed")
        self.assertRegex(result["run_snapshot"], controller.RUN_SNAPSHOT_NAME)
        self.assertTrue(all(item["2"]["inputs"]["value"] == "current" for item in self.queue.prompts))
        for row in result["segments"][2:]:
            self.assertEqual(row["job"]["snapshot_file"], result["run_snapshot"])
            self.assertTrue(row["job"]["generation_graph_fingerprint"])
        for path, original in protected.items():
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), original)
        self.assembly.assert_called_once()

    async def test_completed_project_only_assembles_and_retains_stale_metadata(self):
        for index in range(4):
            self.completed(index, stale=index == 2)
        self.save()
        result = await self.run_generation()
        self.assertEqual(self.queue.indices, [])
        self.assertEqual(result["final_retained_stale_segments"], [2])
        self.assertEqual(result["segments"][2]["job"]["input_fingerprint"], "old-input")
        self.assertEqual(result["segments"][2]["regeneration_reason"], "changed material")

    async def test_stop_and_durable_resume_do_not_rerun_completed_stale_or_queued(self):
        self.completed(0, stale=True)
        self.save()
        def stop(_item):
            plan = self.load()
            plan["stop_requested"] = True
            core.write_plan(self.root, plan)
        self.queue.on_put = stop
        first = await self.run_generation()
        self.assertEqual(first["run_status"], "stopped")
        self.assertEqual(self.queue.indices, [1])
        self.assertEqual(first["run_mode"], "resume_unfinished")
        self.queue.on_put = None
        second = await self.run_generation()
        self.assertEqual(self.queue.indices, [1, 2, 3])
        self.assertEqual(second["segments"][0]["job"]["prompt_id"], "old-0")

    async def test_queued_history_uses_original_saver_when_current_canvas_changed(self):
        self.completed(0, stale=True)
        old_snapshot = snapshot(video="99", value="old queued canvas")
        self.global_snapshot.write_text(json.dumps(old_snapshot), encoding="utf-8")
        self.plan["segments"][1]["job"] = {"status": "queued", "prompt_id": "queued-before"}
        path = self.directory / "old_queued_result.mp4"
        path.write_bytes(b"old queue result")
        self.queue.history["queued-before"] = self.video_history(path, "99")
        self.save()
        original = self.global_snapshot.read_bytes()
        result = await self.run_generation()
        self.assertEqual(self.queue.indices, [2, 3])
        self.assertEqual(result["segments"][1]["job"]["prompt_id"], "queued-before")
        self.assertEqual(result["segments"][1]["job"]["video_id"], "99")
        self.assertEqual(result["segments"][1]["job"]["snapshot_file"], "queue_snapshot.json")
        self.assertEqual(self.global_snapshot.read_bytes(), original)

    async def test_active_queued_job_waits_for_original_result_without_resubmitting(self):
        self.plan["segments"][0]["job"] = {"status": "queued", "prompt_id": "active-before"}
        self.queue.active.add("active-before")
        self.save()
        async def finish(_seconds):
            path = self.directory / "active_result.mp4"
            path.write_bytes(b"active result")
            self.queue.history["active-before"] = self.video_history(path)
            self.queue.active.clear()
        with patch.object(controller.asyncio, "sleep", side_effect=finish):
            result = await self.run_generation()
        self.assertEqual(self.queue.indices, [1, 2, 3])
        self.assertEqual(result["segments"][0]["job"]["prompt_id"], "active-before")

    async def test_old_validation_failure_previous_takes_are_restored_and_skipped(self):
        for index in (0, 1):
            self.completed(index)
            row = self.plan["segments"][index]
            row["replacement_previous_job"] = row.pop("job")
            if index == 1:
                row["job"] = {"status": "failed", "prompt_id": "failed-replacement"}
        self.save()
        protected = self.frozen_files()
        result = await self.run_generation()
        self.assertEqual(self.queue.indices, [2, 3])
        self.assertEqual(result["final_retained_stale_segments"], [0, 1])
        for index in (0, 1):
            row = result["segments"][index]
            self.assertEqual(row["job"]["prompt_id"], f"old-{index}")
            self.assertNotIn("replacement_previous_job", row)
        for name in ("old_0.mp4", "old_1.mp4", "state/queue_snapshot.json"):
            self.assertEqual(self.frozen_files()[name], protected[name])

    def test_previous_take_recovery_with_invalid_video_is_rejected_without_writes(self):
        self.completed(0)
        row = self.plan["segments"][0]
        row["replacement_previous_job"] = row.pop("job")
        row["replacement_previous_job"]["video"] = str(self.directory / "missing.mp4")
        self.save()
        before = self.frozen_files()
        with self.assertRaisesRegex(ValueError, "第 1 段"):
            self.start_without_execute()
        self.assertEqual(self.frozen_files(), before)

    async def test_restart_execute_uses_persisted_mode_and_run_snapshot(self):
        self.completed(0, stale=True)
        self.save()
        self.start_without_execute()
        plan = self.load()
        plan["segments"][1]["job"] = {"status": "queued", "prompt_id": "saved-job",
            "snapshot_file": plan["run_snapshot"], "video_id": "7"}
        path = self.directory / "persisted_result.mp4"
        path.write_bytes(b"persisted result")
        self.queue.history["saved-job"] = self.video_history(path)
        core.write_plan(self.root, plan)
        await controller.execute_project(self.root, self.plan["id"], self.server)
        result = self.load()
        self.assertEqual(self.queue.indices, [2, 3])
        self.assertEqual(result["segments"][0]["job"]["prompt_id"], "old-0")

    def test_retained_video_problems_reject_before_any_write(self):
        for problem in ("missing", "outside", "corrupt", "short", "fps", "dimensions"):
            with self.subTest(problem=problem):
                self.plan = self.load()
                self.completed(0, stale=True)
                self.completed(1)
                if problem == "missing":
                    self.plan["segments"][1]["job"]["video"] = str(self.directory / "missing.mp4")
                elif problem == "outside":
                    self.plan["segments"][1]["job"]["video"] = str(self.output / "outside.mp4")
                self.save()
                before = self.frozen_files()
                good = {"width": 160, "height": 96, "r_frame_rate": "24/1", "nb_read_frames": "24"}
                bad = dict(good)
                if problem == "short": bad["nb_read_frames"] = "23"
                if problem == "fps": bad["r_frame_rate"] = "25/1"
                if problem == "dimensions": bad["width"] = 320
                self.probe.side_effect = [good, RuntimeError("corrupt") if problem == "corrupt" else bad]
                with self.assertRaisesRegex(ValueError, "第 2 段.*单独重新生成"):
                    self.start_without_execute()
                self.assertEqual(self.frozen_files(), before)
                self.assertEqual(self.queue.indices, [])
                self.probe.side_effect = None

    def test_invalid_resume_and_revision_inputs_are_rejected_without_writes(self):
        invalid = [dict(self.payload, resume_unfinished=value) for value in ("true", 1, None, [], {})]
        invalid += [dict(self.payload, revision=value) for value in (True, "3", 3.0, 2, None)]
        invalid += [dict(self.payload, only_segment_index=1),
                    dict(self.payload, resume_unfinished=False, only_segment_index=True),
                    dict(self.payload, resume_unfinished=False, only_segment_index="1")]
        before = self.frozen_files()
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.start_without_execute(payload)
            self.assertEqual(self.frozen_files(), before)

    def test_failed_still_queued_and_unknown_queued_do_not_mutate_project(self):
        for status, active in (("failed", True), ("queued", False)):
            with self.subTest(status=status):
                self.plan["segments"][0]["job"] = {"status": status, "prompt_id": "unknown"}
                self.queue.active = {"unknown"} if active else set()
                self.save()
                before = self.frozen_files()
                with self.assertRaises(ValueError): self.start_without_execute()
                self.assertEqual(self.frozen_files(), before)

    def test_invalid_run_and_job_snapshot_paths_are_rejected(self):
        for name in ("../queue_snapshot.json", "run_snapshot_x.json", "/queue_snapshot.json",
                     "run_snapshot_" + "a" * 32 + ".json/../queue_snapshot.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                controller.run_snapshot_path(self.directory, name)
        self.plan["segments"][0]["job"] = {"status": "queued", "prompt_id": "queued",
                                           "snapshot_file": "../queue_snapshot.json"}
        self.queue.active.add("queued")
        self.save()
        before = self.frozen_files()
        with self.assertRaises(ValueError): self.start_without_execute()
        self.assertEqual(self.frozen_files(), before)

    def test_later_regular_and_single_runs_clear_resume_mode(self):
        self.start_without_execute()
        first_pointer = self.load()["run_snapshot"]
        regular = {**snapshot(), "resume_unfinished": False}
        self.start_without_execute(regular)
        regular_plan = self.load()
        self.assertNotIn("run_mode", regular_plan)
        self.assertNotIn("run_snapshot", regular_plan)
        self.start_without_execute()
        self.start_without_execute({**snapshot(), "only_segment_index": 2, "revision": 3})
        single_plan = self.load()
        self.assertNotIn("run_mode", single_plan)
        self.assertEqual(single_plan["run_only_segment"], 2)
        self.assertNotEqual(single_plan["run_snapshot"], first_pointer)

    async def test_single_middle_segment_stops_and_other_rows_are_unchanged(self):
        self.completed(0, stale=True)
        self.completed(1)
        self.completed(3)
        self.plan.update(final_video=str(self.output / "previous_final.mp4"), final_stale=False)
        self.save()
        before = copy.deepcopy(self.plan["segments"])
        original_global = self.global_snapshot.read_bytes()
        result = await self.run_generation({**snapshot(), "only_segment_index": 1, "revision": 3})
        self.assertEqual(self.queue.indices, [1])
        self.assertEqual(result["run_status"], "paused")
        self.assertNotIn("run_only_segment", result)
        self.assertNotIn("run_mode", result)
        self.assertTrue(result["final_stale"])
        for index in (0, 2, 3): self.assertEqual(result["segments"][index], before[index])
        self.assertEqual(result["segments"][1]["takes"][0], before[1]["job"])
        self.assertEqual(self.global_snapshot.read_bytes(), original_global)
        self.assembly.assert_not_called()

    async def test_single_failed_and_pending_segments_submit_only_target(self):
        for failed in (False, True):
            with self.subTest(failed=failed):
                self.plan = self.load()
                self.plan["segments"][2].pop("job", None)
                if failed:
                    self.plan["segments"][2]["job"] = {"status": "failed", "prompt_id": "failed"}
                self.save()
                self.queue.indices.clear()
                result = await self.run_generation({**snapshot(), "only_segment_index": 2})
                self.assertEqual(self.queue.indices, [2])
                self.assertEqual(result["run_status"], "paused")
                if failed: self.assertEqual(result["segments"][2]["failed_attempts"][0]["prompt_id"], "failed")

    async def test_single_failure_restores_previous_take(self):
        old = self.completed(1)
        old_state = (old.read_bytes(), old.stat().st_mtime_ns)
        self.save()
        self.queue.fail = True
        result = await self.run_generation({**snapshot(), "only_segment_index": 1})
        row = result["segments"][1]
        self.assertEqual(result["run_status"], "failed")
        self.assertEqual(row["job"], self.plan["segments"][1]["job"])
        self.assertTrue(row["needs_regeneration"])
        self.assertEqual(len(row["failed_attempts"]), 1)
        self.assertEqual((old.read_bytes(), old.stat().st_mtime_ns), old_state)

    async def test_workflow_validation_failure_restores_old_take_before_history_stage(self):
        self.completed(1)
        self.save()
        async def invalid(*args): return False, "invalid graph", [], {}
        with patch.object(sys.modules["execution"], "validate_prompt", side_effect=invalid):
            result = await self.run_generation({**snapshot(), "only_segment_index": 1})
        row = result["segments"][1]
        self.assertEqual(self.queue.indices, [])
        self.assertEqual(row["job"], self.plan["segments"][1]["job"])
        self.assertNotIn("replacement_previous_job", row)
        self.assertEqual(result["run_status"], "failed")
        self.assertTrue(row["needs_regeneration"])
        self.assertEqual(len(row["failed_attempts"]), 1)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
    def test_real_ffprobe_rejects_corrupt_short_wrong_fps_and_mismatched_size(self):
        reference = self.directory / "reference.mp4"
        controller.command([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
            "testsrc2=s=160x96:r=24", "-frames:v", "24", "-c:v", "libx264", str(reference)])
        self.plan["segments"][0]["job"] = {"status": "completed", "video": str(reference)}
        for problem, source, frames in (("short", "testsrc2=s=160x96:r=24", 23),
                                        ("fps", "testsrc2=s=160x96:r=25", 25),
                                        ("dimensions", "testsrc2=s=320x96:r=24", 24),
                                        ("corrupt", None, 0)):
            with self.subTest(problem=problem):
                path = self.directory / f"invalid_{problem}.mp4"
                if source is None:
                    path.write_bytes(b"invalid video")
                else:
                    controller.command([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
                        source, "-frames:v", str(frames), "-c:v", "libx264", str(path)])
                self.plan["segments"][1]["job"] = {"status": "completed", "video": str(path)}
                self.save()
                before = self.frozen_files()
                with patch.object(controller, "probe_video", REAL_PROBE), \
                        self.assertRaisesRegex(ValueError, "第 2 段"):
                    self.start_without_execute()
                self.assertEqual(self.frozen_files(), before)
                self.assertEqual(self.queue.indices, [])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
    async def test_real_ffmpeg_completed_resume_assembles_without_changing_old_videos(self):
        source = self.directory / "audio" / "source.wav"
        source.parent.mkdir()
        with wave.open(str(source), "wb") as audio:
            audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(48000)
            audio.writeframes(b"\0\0" * 192000)
        protected = {}
        for index in range(4):
            path = self.directory / f"fixture_{index}.mp4"
            controller.command([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
                "testsrc2=s=160x96:r=24", "-frames:v", "24", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])
            self.plan["segments"][index]["job"] = {"status": "completed", "video": str(path)}
            protected[path] = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        self.plan["segments"][1].update(needs_regeneration=True, regeneration_reason="retained")
        self.save()
        with patch.object(controller, "probe_video", REAL_PROBE), \
                patch.object(controller, "assemble", REAL_ASSEMBLE):
            result = await self.run_generation()
            info = controller.probe_video(result["final_video"])
        self.assertEqual(self.queue.indices, [])
        self.assertEqual(result["run_status"], "completed")
        self.assertEqual(int(info["nb_read_frames"]), 96)
        self.assertEqual(info["r_frame_rate"], "24/1")
        self.assertTrue(result["final_stale"])
        for path, before in protected.items():
            self.assertEqual((hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns), before)


if __name__ == "__main__":
    unittest.main()
