"""Atomic model/optimizer state restore for a declared transformed layout."""

from __future__ import annotations

import torch

from .layouts import (
    StateLayout, _clone, convert_optimizer_state, module_tensors,
    named_optimizer_state, prepare_optimizer_state,
)


def copy_tensors(model, state):
    tensors = module_tensors(model)
    with torch.no_grad():
        for name, value in state.items():
            tensors[name].copy_(value.to(device=tensors[name].device))


class CanonicalStateIO:
    def __init__(self, model, layout: StateLayout, plan_id: str):
        self.model, self.layout, self.plan_id = model, layout, plan_id

    def model_state_dict(self, *, include_nonpersistent=False):
        self.layout.check_module(self.model, transformed=True)
        return self.layout.convert(module_tensors(self.model, persistent_only=not include_nonpersistent),
                                   to_canonical=True, include_nonpersistent=include_nonpersistent)

    def optimizer_state_dict(self, optimizer):
        return convert_optimizer_state(self.layout, named_optimizer_state(self.model, optimizer), to_canonical=True)

    def training_state_dict(self, optimizer, scheduler=None):
        if scheduler is not None and scheduler.optimizer is not optimizer:
            raise ValueError("Scheduler belongs to a different optimizer.")
        return {
            "schema_version": 1, "format": "trainlm_canonical_optimizer",
            "plan_id": self.plan_id, "layout": self.layout.to_dict(),
            "model": self.model_state_dict(), "optimizer": self.optimizer_state_dict(optimizer),
            "nonpersistent_buffers": {spec.name: _clone(module_tensors(self.model)[spec.name])
                                      for spec in self.layout.transformed_tensors if not spec.persistent},
            "optimizer_identity": _optimizer_identity(optimizer),
            "scheduler_class": _class_name(scheduler) if scheduler is not None else None,
            "scheduler": _clone(scheduler.state_dict()) if scheduler is not None else None,
            "cpu_rng": torch.get_rng_state().clone(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        }

    def load_model_state_dict(self, canonical_state):
        self._restore(canonical_state)

    def load_training_state_dict(self, payload, optimizer, scheduler=None):
        if (payload.get("schema_version") != 1 or payload.get("format") != "trainlm_canonical_optimizer"
                or payload.get("plan_id") != self.plan_id
                or StateLayout.from_dict(payload["layout"]) != self.layout):
            raise ValueError("Training state plan/layout identity does not match.")
        if (payload["scheduler"] is None) != (scheduler is None):
            raise ValueError("Resume requires the same scheduler presence.")
        if (payload.get("optimizer_identity") != _optimizer_identity(optimizer)
                or payload.get("scheduler_class") != (_class_name(scheduler) if scheduler is not None else None)):
            raise ValueError("Optimizer/scheduler implementation or state precision changed.")
        if scheduler is not None and scheduler.optimizer is not optimizer:
            raise ValueError("Scheduler belongs to a different optimizer.")
        if payload["cuda_rng"] is not None and not torch.cuda.is_initialized():
            raise ValueError("CUDA RNG restore requires an already initialized CUDA runtime.")
        expected_buffers = {spec.name for spec in self.layout.tensors if not spec.persistent}
        if set(payload["nonpersistent_buffers"]) != expected_buffers:
            raise ValueError("Training state nonpersistent buffer keys changed.")
        self.layout.validate(payload["model"])
        combined = {**payload["model"], **payload["nonpersistent_buffers"]}
        self._restore(combined, include_nonpersistent=True, optimizer=optimizer, optimizer_state=payload["optimizer"],
                      scheduler=scheduler, scheduler_state=payload["scheduler"],
                      rng=(payload["cpu_rng"], payload["cuda_rng"]))

    def _restore(self, canonical_state, *, optimizer=None, optimizer_state=None,
                 scheduler=None, scheduler_state=None, rng=None, include_nonpersistent=False):
        self.layout.check_module(self.model, transformed=True)
        transformed = self.layout.convert(canonical_state, include_nonpersistent=include_nonpersistent)
        prepared = None
        if optimizer is not None:
            if len(optimizer.param_groups) != len(optimizer_state["param_groups"]):
                raise ValueError("Optimizer group count changed.")
            # Validate current ownership too; never overwrite a foreign optimizer.
            current_optimizer = named_optimizer_state(self.model, optimizer)
            converted = convert_optimizer_state(self.layout, optimizer_state)
            if any(set(current["params"]) != set(saved["params"])
                   for current, saved in zip(current_optimizer["param_groups"], converted["param_groups"])):
                raise ValueError("Optimizer group parameter membership changed.")
            prepared = prepare_optimizer_state(self.model, converted)
        before = {name: _clone(value) for name, value in module_tensors(self.model, persistent_only=not include_nonpersistent).items()}
        old_optimizer = (optimizer.state, optimizer.param_groups) if optimizer is not None else None
        old_scheduler = _clone(scheduler.state_dict()) if scheduler is not None else None
        old_cpu = torch.get_rng_state()
        old_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        try:
            copy_tensors(self.model, transformed)
            if prepared is not None:
                optimizer.state, optimizer.param_groups = prepared
            if scheduler is not None:
                scheduler.load_state_dict(scheduler_state)
            if rng is not None:
                torch.set_rng_state(rng[0])
                if rng[1] is not None:
                    torch.cuda.set_rng_state_all(rng[1])
        except BaseException:
            copy_tensors(self.model, before)
            if old_optimizer is not None:
                optimizer.state, optimizer.param_groups = old_optimizer
            if scheduler is not None:
                scheduler.load_state_dict(old_scheduler)
            torch.set_rng_state(old_cpu)
            if old_cuda is not None:
                torch.cuda.set_rng_state_all(old_cuda)
            raise


def _class_name(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _optimizer_identity(optimizer):
    policy = getattr(optimizer, "state_policy", None)
    return {"class": _class_name(optimizer), "state_policy": (
        {name: str(getattr(policy, name)) for name in ("mu_dtype", "nu_dtype")}
        if policy is not None else None
    )}
