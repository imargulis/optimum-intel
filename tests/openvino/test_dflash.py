#  Copyright 2026 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import json
import unittest
from pathlib import Path

import nncf
import openvino as ov
from parameterized import parameterized
from transformers import AutoModelForCausalLM
from utils_tests import MODEL_NAMES

from optimum.exporters.openvino import export_from_model
from optimum.intel.openvino.utils import TemporaryDirectory
from optimum.intel.utils.import_utils import is_transformers_version


class DFlashExportTest(unittest.TestCase):
    def _assert_hidden_state_rt_info_is_valid(self, model):
        def find_output_by_locator(model, locator):
            matches = [op for op in model.get_ops() if op.get_friendly_name() == locator["producer"]]
            if len(matches) != 1:
                raise AssertionError(f"Producer {locator['producer']!r} resolved to {len(matches)} OpenVINO nodes")
            output_index = locator["output_index"]
            if not isinstance(output_index, int) or output_index < 0 or output_index >= len(matches[0].outputs()):
                raise AssertionError(f"Producer {locator['producer']!r} has no output {output_index}")
            return matches[0].output(output_index)

        self.assertTrue(model.has_rt_info(["hidden_states_decoder_layers"]))
        annotation = json.loads(model.get_rt_info()["hidden_states_decoder_layers"].value)
        self.assertIsInstance(annotation, dict)
        self.assertIn("layers", annotation)
        locators = annotation["layers"]
        self.assertTrue(locators)
        self.assertEqual(set(locators), {str(layer_id) for layer_id in range(len(locators))})

        resolved_outputs = set()
        for layer_id in range(len(locators)):
            locator = locators[str(layer_id)]
            self.assertIsInstance(locator, dict)
            self.assertIsInstance(locator.get("producer"), str)
            self.assertIsInstance(locator.get("output_index"), int)
            identity = (locator["producer"], locator["output_index"])
            self.assertNotIn(identity, resolved_outputs)
            find_output_by_locator(model, locator)
            resolved_outputs.add(identity)
        return locators

    @parameterized.expand(("qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe"))
    def test_export_hidden_state_locators_for_representative_decoder_models(self, model_type):
        if model_type in {"qwen3_5", "qwen3_5_moe"} and not (
            is_transformers_version(">=", "5.2.0") and is_transformers_version("<=", "5.2.99")
        ):
            self.skipTest("Qwen3.5 hidden-state locator coverage requires Transformers >= 5.2.0 and <= 5.2.99")

        with TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)
            annotated_dir = tmpdirname / "annotated"
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAMES[model_type])
            export_from_model(
                model=model,
                output=annotated_dir,
                task="text-generation",
                preprocessors=None,
                stateful=False,
            )

            annotated_model = ov.Core().read_model(annotated_dir / "openvino_model.xml")
            self._assert_hidden_state_rt_info_is_valid(annotated_model)

    @staticmethod
    def _tiny_gemma4_text_config(config_class, **extra_config):
        config = {
            "vocab_size": 64,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 2,
            "head_dim": 16,
            "global_head_dim": 16,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 8,
        }
        config.update(extra_config)
        return config_class(**config)

    def _assert_gemma4_text_export_is_annotated(self, model):
        with TemporaryDirectory() as tmpdirname:
            output_dir = Path(tmpdirname)
            export_from_model(
                model=model,
                output=output_dir,
                task="text-generation-with-past",
                preprocessors=None,
                stateful=True,
            )
            annotated_model = ov.Core().read_model(output_dir / "openvino_model.xml")
            self._assert_hidden_state_rt_info_is_valid(annotated_model)

    @unittest.skipUnless(is_transformers_version(">=", "5.5.0"), "Gemma 4 requires Transformers >= 5.5.0")
    def test_export_hidden_state_locators_for_gemma4_text(self):
        import importlib

        transformers = importlib.import_module("transformers")
        config = self._tiny_gemma4_text_config(transformers.Gemma4TextConfig, hidden_size_per_layer_input=0)
        self._assert_gemma4_text_export_is_annotated(transformers.Gemma4ForCausalLM(config))

    @unittest.skipUnless(is_transformers_version(">=", "5.10.0"), "Gemma 4 Unified requires Transformers >= 5.10.0")
    def test_export_hidden_state_locators_for_gemma4_unified_text(self):
        import importlib

        transformers = importlib.import_module("transformers")
        config = self._tiny_gemma4_text_config(transformers.Gemma4UnifiedTextConfig)
        self._assert_gemma4_text_export_is_annotated(transformers.Gemma4UnifiedForCausalLM(config))

    def test_hidden_state_locators_survive_weight_compression(self):
        with TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)
            annotated_dir = tmpdirname / "annotated"
            export_from_model(
                model=AutoModelForCausalLM.from_pretrained(MODEL_NAMES["qwen3"]),
                output=annotated_dir,
                task="text-generation",
                preprocessors=None,
                stateful=False,
            )
            xml_path = annotated_dir / "openvino_model.xml"
            original_model = ov.Core().read_model(xml_path)
            layer_ids = set(self._assert_hidden_state_rt_info_is_valid(original_model))
            for mode, kwargs in (
                (nncf.CompressWeightsMode.INT8_ASYM, {}),
                (nncf.CompressWeightsMode.INT4_ASYM, {"all_layers": True, "group_size": -1}),
            ):
                with self.subTest(mode=mode):
                    compressed_model = nncf.compress_weights(ov.Core().read_model(xml_path), mode=mode, **kwargs)
                    locators = self._assert_hidden_state_rt_info_is_valid(compressed_model)
                    self.assertEqual(set(locators), layer_ids)
