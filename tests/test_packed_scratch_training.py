import torch
from transformers import GPT2Config, GPT2LMHeadModel

from trainlm import PackedBinDataset, TrainLMTrainer, TrainLMTrainingArguments
from trainlm.data.writer import write_packed_binary_shard
from trainlm.training import TrainerCallback


class EvaluationRecorder(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_metrics(self, state, control, metrics):
        del state, control
        if "eval_loss" in metrics:
            self.losses.append(metrics["eval_loss"])


def test_random_init_model_trains_evaluates_checkpoints_and_exports_on_packed_data(
    tmp_path,
):
    data_root = tmp_path / "packed"
    write_packed_binary_shard(
        [[1, 2, 3, 4, 5, 6, 7, 8]],
        data_root / "train",
        shard_id="train",
        vocab_size=16,
    )
    write_packed_binary_shard(
        [[8, 7, 6, 5, 4, 3, 2, 1]],
        data_root / "validation",
        shard_id="validation",
        vocab_size=16,
    )
    train_data = PackedBinDataset.from_directory(
        data_root / "train", sequence_length=4, split="train", seed=17
    )
    eval_data = PackedBinDataset.from_directory(
        data_root / "validation", sequence_length=4, split="validation"
    )
    try:
        config = GPT2Config(
            vocab_size=16,
            n_positions=4,
            n_ctx=4,
            n_embd=8,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            bos_token_id=0,
            eos_token_id=15,
            pad_token_id=15,
            use_cache=False,
        )
        torch.manual_seed(17)
        model = GPT2LMHeadModel(config)
        initial_weights = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        output_dir = tmp_path / "run"
        recorder = EvaluationRecorder()
        trainer = TrainLMTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            callbacks=[recorder],
            args=TrainLMTrainingArguments(
                output_dir=output_dir,
                max_steps=2,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                sequence_length=4,
                accelerator="cpu",
                learning_rate=1e-3,
                weight_decay=0.0,
                lr_scheduler_type="constant",
                logging_steps=1,
                save_steps=1,
                eval_steps=2,
                max_eval_batches=2,
            ),
        )

        state = trainer.train()

        assert state.step == 2
        assert state.tokens_seen == 6
        assert state.loss is not None and torch.isfinite(torch.tensor(state.loss))
        assert len(recorder.losses) == 1
        assert recorder.losses[0] >= 0.0
        assert torch.isfinite(torch.tensor(recorder.losses[0]))
        assert (output_dir / "checkpoint-1" / "trainer_state.pt").is_file()
        assert (output_dir / "checkpoint-2" / "trainer_state.pt").is_file()
        assert any(
            not torch.equal(initial_weights[name], value)
            for name, value in model.state_dict().items()
        )

        export_dir = trainer.save_model(output_dir / "export")
        assert (export_dir / "config.json").is_file()
        assert (export_dir / "model.safetensors").is_file()
        exported = GPT2LMHeadModel.from_pretrained(export_dir)
        for name, expected in model.state_dict().items():
            torch.testing.assert_close(exported.state_dict()[name], expected)
    finally:
        train_data.close()
        eval_data.close()
