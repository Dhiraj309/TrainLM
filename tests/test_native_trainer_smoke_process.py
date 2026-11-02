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
            {"content": "Tiny stories teach language."},
            {"content": "Café characters stay readable."},
            {"content": "Training text defines the vocabulary."},
            {"content": "This row must not enter tokenizer fitting."},
        ]
        tokenizer, document_count = train_tokenizer(
            iter(rows), max_examples=3, vocab_size=300, text_field="content",
        )

        self.assertEqual(document_count, 3)
        self.assertEqual(tokenizer.decode(tokenizer.encode("Café stories")), "Café stories")
        self.assertIsInstance(tokenizer.eos_token_id, int)
        self.assertIsInstance(tokenizer.pad_token_id, int)
        self.assertNotEqual(tokenizer.eos_token_id, tokenizer.pad_token_id)

    def test_scratch_gpt2_architecture_is_built_from_explicit_dimensions(self):
        build_config = runpy.run_path(str(SCRIPT))["_scratch_gpt2_config"]
        config = build_config(
            vocab_size=300, sequence_length=128, embedding_size=256,
            layers=4, attention_heads=8, pad_token_id=0, bos_token_id=2,
            eos_token_id=3,
        )

        self.assertEqual(config.model_type, "gpt2")
        self.assertEqual(config.vocab_size, 300)
        self.assertEqual(config.n_positions, 128)
        self.assertEqual(config.n_embd, 256)
        self.assertEqual(config.n_layer, 4)
        self.assertEqual(config.n_head, 8)

    def test_gradient_checkpointing_helper_manages_cache_without_torch(self):
        class FakeModel:
            def __init__(self):
                self.config = type("Config", (), {"use_cache": True})()
                self.enabled = False

            def gradient_checkpointing_enable(self):
                self.enabled = True

        namespace = runpy.run_path(str(SCRIPT))
        model = FakeModel()
        previous_use_cache = namespace["_enable_gradient_checkpointing"](model)
        self.assertTrue(model.enabled)
        self.assertFalse(model.config.use_cache)
        namespace["_restore_generation_cache"](model, previous_use_cache)
        self.assertTrue(model.config.use_cache)

    def test_gradient_checkpointing_disables_cache_and_runs_backward(self):
        import torch
        from transformers import AutoModelForCausalLM

        namespace = runpy.run_path(str(SCRIPT))
        config = namespace["_scratch_gpt2_config"](
            vocab_size=300, sequence_length=16, embedding_size=32,
            layers=2, attention_heads=4, pad_token_id=1, bos_token_id=2,
            eos_token_id=3,
        )
        model = AutoModelForCausalLM.from_config(config)
        previous_use_cache = namespace["_enable_gradient_checkpointing"](model)

        self.assertTrue(model.is_gradient_checkpointing)
        self.assertFalse(model.config.use_cache)
        input_ids = torch.randint(0, 300, (1, 8))
        output = model(input_ids=input_ids, labels=input_ids)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(model.transformer.wte.weight.grad)
        namespace["_restore_generation_cache"](model, previous_use_cache)
        self.assertTrue(model.config.use_cache)

    def test_from_scratch_tokenizer_preserves_eos_document_boundaries(self):
        class FakeTokenizer:
            eos_token_id = 99

            def __call__(self, texts, *, add_special_tokens, truncation):
                self.assert_options = (add_special_tokens, truncation)
                return {"input_ids": [[ord(text[0])] for text in texts]}

        tokenizer = FakeTokenizer()
        token_documents = runpy.run_path(str(SCRIPT))["_token_documents"]
        rows = list(token_documents(
            [{"content": "a"}, {"content": "b"}, {"content": "c"}],
            tokenizer,
            max_examples=2,
            batch_size=1,
            text_field="content",
        ))

        self.assertEqual(rows, [[ord("a"), 99], [ord("b"), 99]])
        self.assertEqual(tokenizer.assert_options, (False, False))

    def test_from_scratch_tokenizer_rejects_missing_text_column(self):
        class FakeTokenizer:
            eos_token_id = 99

            def __call__(self, texts, *, add_special_tokens, truncation):
                return {"input_ids": [[1] for _ in texts]}

        token_documents = runpy.run_path(str(SCRIPT))["_token_documents"]
        with self.assertRaisesRegex(ValueError, "'content' field"):
            list(token_documents(
                [{"text": "wrong column"}], FakeTokenizer(),
                max_examples=1, text_field="content",
            ))

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
        self.assertIn("--model-embedding-size", result.stdout)
        self.assertIn("--model-layers", result.stdout)
        self.assertIn("--model-attention-heads", result.stdout)
        self.assertIn("--bf16", result.stdout)
        self.assertIn("--gradient-checkpointing", result.stdout)
        self.assertIn("--learning-rate", result.stdout)
        self.assertIn("--weight-decay", result.stdout)
        self.assertIn("--warmup-steps", result.stdout)
        self.assertIn("--seed", result.stdout)
        self.assertIn("--dataset-id", result.stdout)
        self.assertIn("--dataset-revision", result.stdout)
        self.assertIn("--train-split", result.stdout)
        self.assertIn("--validation-split", result.stdout)
        self.assertIn("--text-field", result.stdout)

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

    def test_invalid_gpt2_dimensions_fail_before_torch_import(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--output-dir", "/tmp/unused",
             "--from-scratch", "--model-embedding-size", "768",
             "--model-attention-heads", "10"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("divisible by attention heads", result.stderr)
        self.assertNotIn("No module named 'torch_tpu'", result.stderr)

    def test_scratch_dataset_requires_immutable_revision_before_torch_import(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--output-dir", "/tmp/unused",
             "--from-scratch", "--dataset-revision", "main"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("immutable 40-character commit SHA", result.stderr)
        self.assertNotIn("No module named 'torch_tpu'", result.stderr)

    def test_invalid_optimizer_settings_fail_before_torch_import(self):
        cases = (
            ("--learning-rate", "-1", "learning_rate must be finite and positive"),
            ("--weight-decay", "-0.1", "weight_decay must be finite and non-negative"),
            ("--warmup-steps", "-1", "warmup_steps must be a non-negative integer"),
            ("--warmup-steps", "4", "warmup_steps must not exceed max_steps"),
            ("--seed", "-1", "seed must be a non-negative integer"),
        )
        for flag, value, message in cases:
            with self.subTest(flag=flag):
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), "--output-dir", "/tmp/unused",
                     "--from-scratch", flag, value],
                    capture_output=True, text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertNotIn("No module named 'torch_tpu'", result.stderr)

    def test_scratch_dataset_source_accepts_custom_column_and_distinct_splits(self):
        validate = runpy.run_path(str(SCRIPT))["_validate_scratch_dataset_source"]
        validate(
            dataset_id="org/custom-corpus",
            dataset_revision="a" * 40,
            train_split="train",
            validation_split="validation",
            text_field="content",
        )
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate(
                dataset_id="org/custom-corpus",
                dataset_revision="a" * 40,
                train_split="train",
                validation_split="train",
                text_field="content",
            )

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
        self.assertIn("MODEL_EMBEDDING_SIZE", source)
        self.assertIn("MODEL_LAYERS", source)
        self.assertIn("MODEL_ATTENTION_HEADS", source)
        self.assertIn("USE_BF16 = False", source)
        self.assertIn("command.append(\"--bf16\")", source)
        self.assertIn("USE_GRADIENT_CHECKPOINTING = False", source)
        self.assertIn("command.append(\"--gradient-checkpointing\")", source)
        self.assertIn("LEARNING_RATE = 3e-4", source)
        self.assertIn("WEIGHT_DECAY = 0.1", source)
        self.assertIn("WARMUP_STEPS = None", source)
        self.assertIn("SEED = 42", source)
        self.assertIn("--learning-rate", source)
        self.assertIn("--weight-decay", source)
        self.assertIn("--warmup-steps", source)
        self.assertIn("--seed", source)
        self.assertIn("DATASET_ID", source)
        self.assertIn("DATASET_REVISION", source)
        self.assertIn("TEXT_FIELD", source)
        self.assertIn("--dataset-id", source)
        self.assertIn("--text-field", source)
        self.assertIn("trains a byte-level BPE tokenizer", markdown)
        self.assertIn("builds the GPT-2 config locally", markdown)
        self.assertIn("no pretrained model config, weights, or tokenizer are downloaded", markdown)


if __name__ == "__main__":
    unittest.main()
