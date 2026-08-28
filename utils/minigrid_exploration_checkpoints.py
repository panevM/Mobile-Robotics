"""Central checkpoint lifecycle for modular MiniGrid exploration."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from utils.minigrid_dual_scale_dqn import (
    DualScaleMapDQN,
    load_dual_scale_checkpoint,
    save_dual_scale_checkpoint,
)


class CheckpointManager:
    def __init__(self, output_directory):
        self.output_directory = Path(output_directory)
        self.checkpoint_directory = self.output_directory / "checkpoints"
        self.metrics_directory = self.output_directory / "metrics"
        self.artifact_directory = self.output_directory / "validation_artifacts"

    @property
    def latest_checkpoint_path(self):
        return self.checkpoint_directory / "latest_checkpoint.pth"

    @property
    def best_checkpoint_path(self):
        return self.checkpoint_directory / "best_validation_checkpoint.pth"

    def prepare(self):
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self.metrics_directory.mkdir(parents=True, exist_ok=True)
        self.artifact_directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_policy(path, device):
        return load_dual_scale_checkpoint(path, device)

    @staticmethod
    def load_inherited_policy(path, device):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        policy = DualScaleMapDQN(**checkpoint["architecture"]).to(device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        return policy, checkpoint

    def resolve_continuation_checkpoint(self, selection):
        if isinstance(selection, str) and selection in {"latest", "best"}:
            path = (
                self.latest_checkpoint_path
                if selection == "latest"
                else self.best_checkpoint_path
            )
            legacy_name = (
                "latest_checkpoint.pth"
                if selection == "latest"
                else "best_validation_checkpoint.pth"
            )
            legacy_path = self.output_directory / legacy_name
            if not path.exists() and legacy_path.exists():
                path = legacy_path
        else:
            path = Path(selection)
        if not path.exists():
            raise FileNotFoundError(f"continuation checkpoint does not exist: {path.resolve()}")
        return path

    def save(
        self,
        *,
        policy,
        target,
        optimizer,
        metadata,
        episode=None,
        final=False,
        is_best=False,
    ):
        self.prepare()
        if final:
            path = self.output_directory / "final_model.pth"
        elif episode is not None:
            path = self.checkpoint_directory / f"checkpoint_ep_{episode:04d}.pth"
        else:
            raise ValueError("periodic checkpoints require an episode number")
        save_dual_scale_checkpoint(path, policy, target_net=target, optimizer=optimizer, metadata=metadata)
        shutil.copy2(path, self.latest_checkpoint_path)
        # Keep the original root-level alias so existing notebooks continue to work.
        shutil.copy2(path, self.output_directory / "latest_checkpoint.pth")
        if is_best:
            shutil.copy2(path, self.best_checkpoint_path)
        return path


__all__ = ["CheckpointManager"]
