"""Unit tests for FutureTrainer and TrainingHistory orchestration."""

from pathlib import Path
import pytest
import torch
from torch import nn

from src.data.dataset import SyntheticMRIDataset
from src.data.dataloader import create_mri_dataloader
from src.training.config import TrainingConfig
from src.training.trainer import FutureTrainer, TrainingHistory


class MinimalTestModel(nn.Module):
    """Small linear model matching (B, 3, 224, 224) input and (B, 3) output."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)
        return self.fc(pooled)


class TestTrainingHistory:

    def test_record_and_serialize(self, tmp_path):
        history = TrainingHistory()
        history.record(epoch=1, train_loss=0.8, val_loss=0.7, val_acc=0.65, val_f1=0.63)
        history.record(epoch=2, train_loss=0.6, val_loss=0.5, val_acc=0.75, val_f1=0.74)

        d = history.to_dict()
        assert d["epochs"] == [1, 2]
        assert d["train_loss"] == [0.8, 0.6]

        save_path = tmp_path / "history.json"
        history.save(save_path)
        assert save_path.is_file()


class TestFutureTrainer:

    @pytest.fixture
    def trainer_setup(self):
        model = MinimalTestModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = TrainingConfig()
        trainer = FutureTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            device="cpu",
        )
        train_ds = SyntheticMRIDataset(num_samples=8, seed=1)
        val_ds = SyntheticMRIDataset(num_samples=4, seed=2)
        train_loader = create_mri_dataloader(train_ds, batch_size=4, shuffle=True)
        val_loader = create_mri_dataloader(val_ds, batch_size=4, shuffle=False)
        return trainer, train_loader, val_loader

    def test_train_epoch(self, trainer_setup):
        trainer, train_loader, _ = trainer_setup
        loss = trainer.train_epoch(train_loader)
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_validate_epoch(self, trainer_setup):
        trainer, _, val_loader = trainer_setup
        loss, metrics = trainer.validate_epoch(val_loader)
        assert loss is not None and loss > 0.0
        assert metrics.total_samples == 4

    def test_fit_multi_epoch(self, trainer_setup):
        trainer, train_loader, val_loader = trainer_setup
        history = trainer.fit(train_loader, val_loader, epochs=2)
        assert len(history.epochs) == 2
        assert len(history.train_loss) == 2
        assert len(history.val_loss) == 2

    def test_checkpoint_roundtrip(self, trainer_setup, tmp_path):
        trainer, train_loader, _ = trainer_setup
        trainer.train_epoch(train_loader)

        ckpt_path = tmp_path / "checkpoint.pt"
        saved = trainer.save_checkpoint(ckpt_path, epoch=1, metrics={"test_acc": 0.8})
        assert saved.is_file()

        # Load into fresh trainer
        new_model = MinimalTestModel()
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
        new_trainer = FutureTrainer(
            model=new_model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=new_optimizer,
            device="cpu",
        )
        payload = new_trainer.load_checkpoint(ckpt_path)
        assert payload["epoch"] == 1
        assert payload["metrics"]["test_acc"] == 0.8
