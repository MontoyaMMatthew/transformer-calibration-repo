"""Lightweight portability checks: no model, CUDA workload, or downloads."""

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import setup_common
from src import artifacts, inference, run_experiments


class PortabilityTests(unittest.TestCase):
    def test_storage_root_is_required_and_custom_paths_are_respected(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Set TRANSFORMER"):
                artifacts.persistent_path("artifacts/run")
            os.environ["TRANSFORMER_CALIBRATION_PERSISTENT_ROOT"] = "relative"
            with self.assertRaisesRegex(ValueError, "absolute"):
                setup_common.configure_paths()
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                os.environ["TRANSFORMER_CALIBRATION_PERSISTENT_ROOT"] = str(root)
                self.assertEqual(artifacts.persistent_path(root / "results"), root / "results")
                with self.assertRaisesRegex(ValueError, "persistent root"):
                    artifacts.persistent_path(root.parent / "outside")
                with patch.object(setup_common, "ROOT", root):
                    paths = setup_common.configure_paths()
                self.assertTrue(all(p.is_relative_to(root) for p in paths.values()))

    def test_extracted_source_can_be_sealed_without_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "src" / "example.py"
            source.parent.mkdir()
            source.write_text("# Example source snapshot\n")
            run = root / "run"
            with (patch.object(setup_common, "ROOT", root),
                  patch.object(setup_common.subprocess, "run", side_effect=FileNotFoundError("git")),
                  patch.object(artifacts, "ROOT", root),
                  patch.object(artifacts, "code_files", return_value=[source]),
                  patch.object(artifacts, "environment", return_value={"sources": "snapshot"}),
                  patch.object(artifacts, "configure_paths", return_value={})):
                config = {"batch_size": 1, "purpose": "Synthetic portability check"}
                manifest = artifacts.save_manifest(run, config, [], {}, {}, {}, {})
                self.assertIsNone(manifest["git_revision"])
                self.assertIsNone(manifest["git_status"])
                self.assertEqual((run / "source/src/example.py").read_bytes(), source.read_bytes())
                self.assertEqual(artifacts.load_run(run, check_environment=True)[1], config)

    def test_gpu_capabilities_replace_name_filter(self):
        torch = types.SimpleNamespace(
            __version__="mock",
            version=types.SimpleNamespace(cuda="11.8", hip=None),
            cuda=types.SimpleNamespace(
                is_available=Mock(return_value=True), device_count=Mock(return_value=1),
                get_device_properties=Mock(return_value=types.SimpleNamespace(
                    name="A different NVIDIA GPU", total_memory=24 * 1024**3)),
                is_bf16_supported=Mock(return_value=True),
                get_device_capability=Mock(return_value=(8, 0))),
            backends=types.SimpleNamespace(cuda=types.SimpleNamespace(
                is_flash_attention_available=Mock(return_value=True))))
        config = {"minimum_vram_gib": 16.2}
        with (patch.dict(sys.modules, {"torch": torch}),
              patch.object(inference.subprocess, "run", return_value=types.SimpleNamespace(stdout="GPU info"))):
            self.assertEqual(inference.hardware(config)["name"], "A different NVIDIA GPU")
            torch.cuda.is_bf16_supported.assert_called_with(including_emulation=False)
            torch.cuda.get_device_properties.return_value.total_memory = 16 * 1024**3
            with self.assertRaisesRegex(RuntimeError, "16.2 GiB"):
                inference.hardware(config)
            torch.cuda.get_device_properties.return_value.total_memory = 24 * 1024**3
            torch.cuda.is_bf16_supported.return_value = False
            with self.assertRaisesRegex(RuntimeError, "Native hardware BF16"):
                inference.hardware(config)
            torch.cuda.is_bf16_supported.return_value = True
            torch.backends.cuda.is_flash_attention_available.return_value = False
            with self.assertRaisesRegex(RuntimeError, "Flash SDPA"):
                inference.hardware(config)

    def test_pilot_defaults_and_memory_policy(self):
        config = json.loads((setup_common.ROOT / "config/pilot.json").read_text())
        self.assertNotIn("expected_gpu", config)
        self.assertEqual(config["batch_size"], 1)
        self.assertEqual(config["minimum_vram_gib"], 16.2)
        self.assertNotIn("maximum_memory_fraction", config)
        run_experiments.validate_config(config)
        config["minimum_vram_gib"] = 0
        with self.assertRaises(ValueError):
            run_experiments.validate_config(config)

    def test_execution_records_headroom_without_percentage_cutoff(self):
        config = json.loads((setup_common.ROOT / "config/pilot.json").read_text())
        case = {"evaluation_id": "mock-evaluation", "context_budget": 2048, "input_tokens": 10}
        transformers = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
            from_pretrained=Mock(return_value=object())))
        for headroom in (0.11, 0.09, 0.01):
            with self.subTest(headroom=headroom), tempfile.TemporaryDirectory() as directory:
                class FakeScorer:
                    load_seconds = 0.0
                    def __init__(self, *args):
                        pass
                    def attention_check(self, *args):
                        return {"status": "mock"}
                    def score(self, cases, prompts):
                        return [{**c, "minimum_headroom_fraction": headroom} for c in cases]

                with (patch.dict(sys.modules, {"transformers": transformers}),
                      patch.object(inference, "hardware", return_value={"mock": True}),
                      patch.object(inference, "Scorer", FakeScorer),
                      patch.object(run_experiments, "load_run", return_value=({"identity": "mock"}, config, [case])),
                      patch.object(run_experiments, "load_records", return_value={}),
                      patch.object(run_experiments, "read", side_effect=lambda p:
                                   config if p == setup_common.ROOT / "config/pilot.json"
                                   else {"model": {"snapshot_path": "mock"}}),
                      patch.object(run_experiments, "reconstruct", return_value="mock prompt"),
                      patch.object(run_experiments, "verify_record")):
                    run_experiments.execute(Path(directory))
                    saved = json.loads((Path(directory) / "records/mock-evaluation.json").read_text())
                    self.assertEqual(saved["minimum_headroom_fraction"], headroom)

    def test_package_cli_uses_custom_root_and_verifies_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run = root / "example-run"
            run.mkdir()
            (run / "result.json").write_text('{"saved": true}\n')
            output = root / "backups/example.tar.gz"
            command = [sys.executable, str(setup_common.ROOT / "scripts/package_run.py"),
                       "--include", str(run), "--output", str(output)]
            env = dict(os.environ, TRANSFORMER_CALIBRATION_PERSISTENT_ROOT=str(root))
            result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
            self.assertIn("Verified 1 members", result.stdout)
            with tarfile.open(output) as archive:
                self.assertEqual(archive.extractfile("example-run/result.json").read(),
                                 (run / "result.json").read_bytes())
            self.assertTrue(Path(str(output) + ".sha256").is_file())
            self.assertNotEqual(subprocess.run(command, env=env, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
