"""The trainer smoke launcher must remain usable before Torch is installed."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/torchtpu_native_trainer_smoke.py"
NOTEBOOK = ROOT / "notebooks/TrainLM_Native_TorchTPU_Single_Device_Training.ipynb"


class NativeTrainerSmokeProcessTests(unittest.TestCase):
    def test_scratch_tokenizer_is_trained_from_bounded_training_text(self):
        train_tokenizer = runpy.run_path(str(SCRIPT))["_train_byte_level_tokenizer"]
        rows = [
            {"text": "Tiny stories teach language."},
            {"text": "Café characters stay readable."},
            {"text": "Training text defines the vocabulary."},
            {"text": "This row must not enter tokenizer fitting."},
        ]
        tokenizer, document_count = train_tokenizer(
            iter(rows), max_examples=3, vocab_size=300,
        )

        self.assertEqual(document_count, 3)
        self.assertEqual(tokenizer.decode(tokenizer.encode("Café stories")), "Café stories")
        self.assertIsInstance(tokenizer.eos_token_id, int)
        self.assertIsInstance(tokenizer.pad_token_id, int)
        self.assertNotEqual(tokenizer.eos_token_id, tokenizer.pad_token_id)

    def test_from_scratch_tokenizer_preserves_eos_document_boundaries(self):
        class FakeTokenizer:
            eos_token_id = 99

            def __call__(self, texts, *, add_special_tokens, truncation):
                self.assert_options = (add_special_tokens, truncation)
                return {"input_ids": [[ord(text[0])] for text in texts]}

        tokenizer = FakeTokenizer()
        token_documents = runpy.run_path(str(SCRIPT))["_token_documents"]
        rows = list(token_documents(
            [{"text": "a"}, {"text": "b"}, {"text": "c"}],
            tokenizer,
            max_examples=2,
            batch_size=1,
        ))

        self.assertEqual(rows, [[ord("a"), 99], [ord("b"), 99]])
        self.assertEqual(tokenizer.assert_options, (False, False))

    def test_help_does_not_initialize_torch(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("--resume", result.stdout)
        self.assertIn("--compile", result.stdout)
        self.assertIn("--from-scratch", result.stdout)
        self.assertIn("--train-examples", result.stdout)
        self.assertIn("--tokenizer-vocab-size", result.stdout)

    def test_incompatible_modes_fail_before_torch_import(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--output-dir", "/tmp/unused",
             "--compile", "--resume"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Run checkpoint/resume after the eager baseline", result.stderr)
        self.assertNotIn("No module named 'torch_tpu'", result.stderr)

    def test_invalid_scratch_step_count_fails_before_torch_import(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--output-dir", "/tmp/unused",
             "--from-scratch", "--max-steps", "0"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be positive", result.stderr)
        self.assertNotIn("No module named 'torch_tpu'", result.stderr)

    def test_invalid_tokenizer_size_fails_before_torch_import(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--output-dir", "/tmp/unused",
             "--from-scratch", "--tokenizer-vocab-size", "259"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("vocab_size must be at least 260", result.stderr)
        self.assertNotIn("No module named 'torch_tpu'", result.stderr)

    def test_notebook_keeps_device_import_inside_bounded_child(self):
        notebook = json.loads(NOTEBOOK.read_text())
        cells = ["".join(cell["source"]) for cell in notebook["cells"]
                 if cell["cell_type"] == "code"]
        source = "\n".join(cells)
        self.assertNotIn("import torch_tpu", source)
        self.assertNotIn("TrainLMTrainer(", source)
        self.assertIn("run_probe_process", source)
        self.assertIn("torchtpu_native_trainer_smoke.py", source)

    def test_scratch_notebook_trains_tokenizer_on_the_train_split(self):
        notebook = json.loads(
            (ROOT / "notebooks/TrainLM_Native_TorchTPU_From_Scratch.ipynb").read_text()
        )
        source = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        markdown = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        self.assertIn("TOKENIZER_VOCAB_SIZE", source)
        self.assertIn("--tokenizer-vocab-size", source)
        self.assertIn("trains a byte-level BPE tokenizer", markdown)
        self.assertIn(
            "pretrained model weights and tokenizer are never downloaded",
            markdown,
        )


if __name__ == "__main__":
    unittest.main()
