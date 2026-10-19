# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji-specific extension of mjlab's on-policy runner.

Restricts W&B artifact uploads to the final-iteration checkpoint only.
Local ``.pt`` files are still written on every ``save_interval`` boundary
(disk-only, useful for crash recovery); W&B receives just the terminal
``model_{N-1}.pt``.

Subclasses that emit their own artifacts (e.g. an ``.onnx`` export
alongside the ``.pt``) must call :meth:`WujiOnPolicyRunner.should_upload_artifact`
before invoking ``self.logger.save_model`` so they honor the same gate.
"""

from __future__ import annotations

from mjlab.rl import MjlabOnPolicyRunner


class WujiOnPolicyRunner(MjlabOnPolicyRunner):
  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    self._final_iteration = (
      self.current_learning_iteration + num_learning_iterations - 1
    )
    super().learn(num_learning_iterations, init_at_random_ep_len)

  def save(self, path: str, infos=None) -> None:
    # Temporarily flip cfg["upload_model"] so the parent's save() handles
    # the W&B upload conditionally. This avoids re-implementing the parent's
    # local-checkpoint logic, which is fragile against upstream mjlab changes.
    original = self.cfg.get("upload_model")
    self.cfg["upload_model"] = self.should_upload_artifact()
    try:
      super().save(path, infos)
    finally:
      self.cfg["upload_model"] = original

  def should_upload_artifact(self) -> bool:
    """Whether the current iteration is eligible for a W&B artifact upload.

    True iff ``upload_model`` is set in the cfg, ``learn()`` has been
    called (so ``_final_iteration`` is known), and the current iteration
    has reached the final one of the training session. The comparison
    uses ``>=`` (not ``==``) so a one-off counter shift in upstream rsl-rl
    — e.g. the loop incrementing ``current_learning_iteration`` past
    ``N-1`` before the final ``save()`` — surfaces as an extra upload
    rather than a silently-dropped final ckpt.

    Subclasses that emit their own artifacts (e.g. ONNX exports) should
    gate ``self.logger.save_model`` calls on this predicate.
    """
    if not self.cfg.get("upload_model"):
      return False
    final = getattr(self, "_final_iteration", None)
    if final is None:
      return False
    return self.current_learning_iteration >= final
