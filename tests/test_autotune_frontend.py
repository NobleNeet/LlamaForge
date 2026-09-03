import os
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))


class TestAutoTuneFrontendBoundary(unittest.TestCase):
    def test_models_owns_editor_and_autotune_has_no_circular_import(self):
        with open(os.path.join(ROOT, "web", "js", "models.js"), encoding="utf-8") as handle: models = handle.read()
        with open(os.path.join(ROOT, "web", "js", "autotune.js"), encoding="utf-8") as handle: autotune = handle.read()
        self.assertIn('from "./autotune.js"', models)
        self.assertNotIn('from "./models.js"', autotune)
        self.assertIn("data-autotune-panel", models)
        self.assertIn("bridge.stage", autotune)

    def test_autotune_live_region_is_separate_from_knob_grid(self):
        with open(os.path.join(ROOT, "web", "js", "models.js"), encoding="utf-8") as handle: models = handle.read()
        self.assertLess(models.index('class="ed-autotune"'), models.index('class="ed-knobs"'))
        self.assertIn("syncAutoTune(m)", models)

    def test_stage_local_progress_has_no_overall_percentage(self):
        with open(os.path.join(ROOT, "web", "js", "autotune.js"), encoding="utf-8") as handle: autotune = handle.read()
        self.assertIn("Stage ${stageNumber} / ${esc(stageCount)}", autotune)
        self.assertIn("cases in this stage", autotune)
        self.assertIn("Waiting for benchmark resource...", autotune)
        self.assertIn("stageHistory(progress)", autotune)
        self.assertNotIn("overall percentage", autotune)

    def test_stage_counts_use_an_aligned_left_side_grid(self):
        with open(os.path.join(ROOT, "web", "index.html"), encoding="utf-8") as handle: html = handle.read()
        stage_css = html[html.index(".at-stage{"):html.index("/* modal", html.index(".at-stage{"))]
        self.assertIn("display:grid", stage_css)
        self.assertIn("grid-template-columns:10px minmax(100px,130px)", stage_css)
        self.assertIn("margin-left:0", stage_css)
        self.assertNotIn("margin-left:auto", stage_css)

    def test_completed_run_reference_is_restored_until_rerun(self):
        with open(os.path.join(ROOT, "web", "js", "autotune.js"), encoding="utf-8") as handle: autotune = handle.read()
        self.assertIn('RUN_STORAGE_KEY = "lf_autotune_runs_v1"', autotune)
        self.assertIn("saveRun(modelId, s.runId)", autotune)
        self.assertIn("if (s.runId && !s.restored)", autotune)
        self.assertIn("data-autotune-rerun", autotune)
        self.assertIn('"Run again"', autotune)

    def test_strategy_v2_stage_labels_are_present(self):
        with open(os.path.join(ROOT, "web", "js", "autotune.js"), encoding="utf-8") as handle: autotune = handle.read()
        for label in ("Batch sizing", "Flash attention", "KV cache", "Final validation"):
            self.assertIn(label, autotune)

    def test_failed_stage_and_not_run_stages_are_rendered(self):
        with open(os.path.join(ROOT, "web", "js", "autotune.js"), encoding="utf-8") as handle: autotune = handle.read()
        self.assertIn('item.status === "failed" ? "✕"', autotune)
        self.assertIn('item.status === "not_run"', autotune)
        self.assertIn('terminal.has(status)', autotune)

    def test_preset_save_uses_modal_instead_of_native_prompt(self):
        with open(os.path.join(ROOT, "web", "js", "models.js"), encoding="utf-8") as handle: models = handle.read()
        self.assertIn('showModal("Save preset"', models)
        self.assertIn('data-preset-save-form', models)
        self.assertIn('confirmPresetSave(', models)
        self.assertNotIn('prompt("Preset name', models)

    def test_preset_bar_buttons_are_explicit_non_submit_buttons(self):
        with open(os.path.join(ROOT, "web", "js", "models.js"), encoding="utf-8") as handle: models = handle.read()
        self.assertIn('type="button" data-preset-save', models)
        self.assertIn('type="button" data-preset-overwrite', models)

    def test_preset_actions_are_bound_to_the_rendered_row(self):
        with open(os.path.join(ROOT, "web", "js", "models.js"), encoding="utf-8") as handle: models = handle.read()
        self.assertIn("function wirePresetActions(row)", models)
        self.assertIn("wirePresetActions(row);", models)
        self.assertIn("await savePresetFrom(save.dataset.presetSave, row)", models)
        self.assertIn("overwritePresetFrom(overwrite.dataset.presetOverwrite,", models)

    def test_overwrite_uses_the_app_modal_instead_of_native_confirm(self):
        with open(os.path.join(ROOT, "web", "js", "models.js"), encoding="utf-8") as handle: models = handle.read()
        self.assertIn('showModal("Overwrite bound preset"', models)
        self.assertIn('id="preset-overwrite-confirm"', models)
        self.assertNotIn('confirm(`Overwrite preset', models)
