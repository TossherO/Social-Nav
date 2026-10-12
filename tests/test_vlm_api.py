"""Portable remote VLM regressions: synthetic assets and mocked HTTP only."""
import base64
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import httpx

from decision.vlm import benchmark_vlm as api
from decision.vlm import package_vlm_test as pack
from decision.vlm import benchmark_defaults as defaults
from decision.vlm.jrdb_t1_inputs import rel_schema


MODEL = "qwen3.8-27b-fp8"


def fixture(directory, count=3):
    directory = Path(directory)
    directory.mkdir(parents=True)
    (directory / "system_prompt.txt").write_text("Return typed REL JSON", encoding="utf-8")
    cases, image_bytes = [], 0
    for i in range(count):
        folder = directory / "cases" / f"case_{i:02d}"
        folder.mkdir(parents=True)
        images = []
        for j in range(3):
            path = folder / f"image_{j}.png"
            path.write_bytes(f"SYNTHETIC_IMAGE_{i}_{j}".encode())
            image_bytes += path.stat().st_size
            images.append(path.relative_to(directory).as_posix())
        model = {"wire_protocol": "object", "images": images,
                 "fields": {"v": [0, 1, 2, 3], "dt": [-1, -.5, 0], "ts": "fixture",
                            "e": [-2, 12, -6, 6], "o": [], "d": []},
                 "user_prompt": f"JSON observations {i}",
                 "response_schema": rel_schema([0, 1, 2, 3])}
        path = folder / "model_input.json"
        path.write_text(json.dumps(model), encoding="utf-8")
        cases.append({"case_id": f"case_{i:02d}", "scene": ["clark", "gates", "memorial"][i % 3],
                      "input_file": path.relative_to(directory).as_posix()})
    files = {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in directory.rglob("*") if p.is_file()}
    manifest = {"native_social_labels_included": False,
                "credentials_included": False, "prompt_profile": "object", "wire_protocol": "object",
                "profile": "synthetic", "image_bytes_total": image_bytes, "files": files, "cases": cases}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return api.common.load_bundle(directory)


def stream(text='{"g":[{"m":[0,1],"r":"w","k":0}],"u":[]}', *,
           model=MODEL, done=True, finish="stop", reasoning=False, reasoning_tokens=0):
    events = [{"model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}}]}]
    if reasoning:
        events.append({"model": model, "choices": [{"index": 0,
                       "delta": {"reasoning_content": "SYNTHETIC_HIDDEN_REASONING"}}]})
    events.extend([
        {"model": model, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}]},
        {"model": model, "choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20,
         "total_tokens": 120, "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
         "private": "DO_NOT_LOG"}}])
    payload = "\n\n".join("data: " + json.dumps(e) for e in events)
    if done:
        payload += "\n\ndata: [DONE]\n\n"
    return httpx.Response(200, text=payload)


class VLMTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = self.root / "bundle"
        self.manifest, self.system, cases = fixture(self.bundle)
        self.case = cases[0]
        self.key = "SYNTHETIC_SERVER_TOKEN_NOT_REAL"
        self.config_path = self.root / "server_config.json"
        self.config_path.write_text(json.dumps({"base_url": "https://my-gpu.example/v1",
                                                "api_key": self.key, "model": ""}))
        self.config = api.load_config(self.config_path)

    def execute(self, response):
        with httpx.Client(transport=httpx.MockTransport(lambda request: response)) as client:
            return api.execute(client, self.config, self.system, self.case,
                               MODEL, "json-schema", 1024, 1)[0]

    def run_mock(self, handler, flags=(), **patches):
        args = api.arguments(["--bundle", str(self.bundle), "--config", str(self.config_path),
                              "--out", str(self.root / "results"), *flags])
        client = httpx.Client(transport=httpx.MockTransport(handler))
        with contextlib.ExitStack() as stack:
            factory = stack.enter_context(patch.object(api.httpx, "Client", return_value=client))
            for name, value in patches.items():
                stack.enter_context(patch.object(api, name, value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            result = api.run(args)
        return result, factory

    def test_defaults_and_request_preserve_assets_with_sglang_controls(self):
        args = api.arguments([])
        self.assertEqual((args.limit, args.warmups, args.constraint), (3, 0, "json-schema"))
        body = api.make_body(self.system, self.case, MODEL)
        original = api.common.make_body(self.system, self.case, constraint="json-schema")
        self.assertEqual(body["messages"], original["messages"])
        self.assertEqual(body["response_format"], original["response_format"])
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(body["model"], MODEL)
        self.assertNotIn("enable_thinking", body)
        self.assertNotIn("max_completion_tokens", body)
        self.assertEqual(body["max_tokens"], 1024)
        images = [p["image_url"]["url"] for p in body["messages"][1]["content"] if p["type"] == "image_url"]
        self.assertEqual(len(images), 3)
        for path, encoded in zip(self.case["paths"], images):
            self.assertEqual(path.read_bytes(), base64.b64decode(encoded.split(",", 1)[1]))

    def test_configuration_accepts_https_and_loopback_not_cloud_or_unsafe_urls(self):
        for base, expected in (("https://my-gpu.example/", "https://my-gpu.example/v1"),
                               ("http://127.0.0.1:6008/v1", "http://127.0.0.1:6008/v1"),
                               ("http://[::1]:6008", "http://[::1]:6008/v1")):
            with self.subTest(base=base):
                self.config_path.write_text(json.dumps({"base_url": base}))
                config = api.load_config(self.config_path)
                self.assertEqual(config["base_url"], expected)
                self.assertNotIn("Authorization", api.headers(config))
        for base in ("", "http://public.example:6008", "https://u:p@my-gpu.example/v1",
                     "https://my-gpu.example/v1?token=x", "https://my-gpu.example/v1#x",
                     "https://my-gpu.example/v1/chat/completions", "https://my-gpu.example:0",
                     "https://maas.qianwenaiapi.com/compatible-mode/v1", "https://api.openai.com/v1"):
            with self.subTest(base=base):
                self.config_path.write_text(json.dumps({"base_url": base}))
                with self.assertRaises(ValueError):
                    api.load_config(self.config_path)
        self.config_path.write_text(json.dumps({"base_url": "http://public.example:6008",
                                                "allow_insecure_http": True}))
        self.assertEqual(api.load_config(self.config_path)["base_url"], "http://public.example:6008/v1")

    def test_model_discovery_uses_exact_service_alias_and_requires_unambiguous_choice(self):
        for model in (MODEL, "/root/models/qwen38-27b", "Qwen/Qwen3.8-27B-FP8"):
            self.assertEqual(api.choose_model({"data": [{"id": model}]})[0], model)
        data = {"data": [{"id": MODEL}, {"id": "qwen3.8-27b"}]}
        with self.assertRaises(ValueError):
            api.choose_model(data)
        self.assertEqual(api.choose_model(data, MODEL)[0], MODEL)
        for data in ({"data": []}, {"data": [{"id": "qwen3.8-flash"}]}, [], {"data": "bad"}):
            with self.assertRaises(ValueError):
                api.choose_model(data)

    def test_check_failure_is_get_only_without_logging_error_body_or_token(self):
        for response in (httpx.Response(401, text=self.key), httpx.Response(302, headers={"Location": "https://bad.example"}),
                         httpx.Response(200, text="<html>not API</html>")):
            seen = []
            def handler(request):
                seen.append(request.method)
                return response
            with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
                result = api.check_server(client, self.config)
            self.assertEqual(seen, ["GET"])
            self.assertEqual(result["status"], "check_failed")
            self.assertNotIn(self.key, json.dumps(result))

    def test_optional_server_info_keeps_only_whitelist_and_can_be_unavailable(self):
        def handler(request):
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": MODEL}]})
            return httpx.Response(200, json={"version": "synthetic", "api_key": self.key,
                "server_args": {"context_length": 8192, "speculative_algorithm": "DSPARK", "api_key": self.key}})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result = api.check_server(client, self.config, info=True)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["server_args"], {"context_length": 8192, "speculative_algorithm": "DSPARK"})
        self.assertNotIn(self.key, json.dumps(result))
        for failure in (httpx.Response(404), httpx.ReadTimeout(self.key)):
            def optional_handler(request):
                if request.url.path.endswith("/models"):
                    return httpx.Response(200, json={"data": [{"id": MODEL}]})
                if isinstance(failure, Exception):
                    raise failure
                return failure
            with httpx.Client(transport=httpx.MockTransport(optional_handler)) as client:
                result = api.check_server(client, self.config, info=True)
            self.assertEqual(result["status"], "ready")
            self.assertNotIn(self.key, json.dumps(result))

    def test_malformed_stream_is_logged_as_one_failure_without_crashing(self):
        row = self.execute(httpx.Response(200, text='data: {"choices":["bad-shape"]}\n\n'))
        self.assertFalse(row["valid"])
        self.assertEqual(row["error_type"], "AttributeError")
        self.assertEqual(row["failure_stage"], "stream")

    def test_valid_complete_response_and_lossless_role_order(self):
        row = self.execute(stream('{"g":[{"m":[1,0,3,2],"r":"p","k":1}],"u":[]}'))
        self.assertTrue(row["valid"])
        self.assertFalse(row["server_hard_schema_verified"])
        self.assertEqual(row["canonical_rel"]["g"][0], {"m": [1, 0, 2, 3], "r": "p", "k": 1})
        self.assertIn("first_output_text_s", row["timing"])
        self.assertNotIn("cached_tokens", row["usage"])
        self.assertNotIn("private", row["usage"])

    def test_invalid_membership_types_flags_and_unknown_properties_not_repaired(self):
        for text in ('{"g":[{"m":["0",1],"r":"w","k":0}],"u":[]}',
                     '{"g":[{"m":[0.0,1],"r":"w","k":0}],"u":[]}',
                     '{"g":[{"m":[0,1],"r":"w","k":0}],"u":[1]}',
                     '{"g":[{"m":[0,9],"r":"w","k":0}],"u":[]}',
                     '{"g":[{"m":[0,1],"r":"w","k":1}],"u":[]}',
                     '{"g":[{"m":[0,1],"r":"w","k":0,"extra":1}],"u":[]}',
                     '{"g":[[[0,1],"w",0]],"u":[]}'):
            with self.subTest(text=text):
                row = self.execute(stream(text))
                self.assertFalse(row["valid"])
                self.assertNotIn("canonical_rel", row)

    def test_reasoning_is_rejected_and_not_logged_in_any_form(self):
        for response in (stream(reasoning=True), stream(reasoning_tokens=5),
                         stream('<think>SYNTHETIC_HIDDEN_REASONING</think>{"g":[],"u":[]}')):
            row = self.execute(response)
            self.assertFalse(row["valid"])
            self.assertTrue(row["reasoning_content_present"])
            self.assertNotIn("raw_content", row)
            self.assertNotIn("SYNTHETIC_HIDDEN_REASONING", json.dumps(row))

    def test_truncation_model_mismatch_and_errors_rejected(self):
        for response in (stream(done=False), stream(finish="length"), stream("not JSON"),
                         stream(model="qwen3.8-flash"), httpx.Response(400, text=self.key)):
            row = self.execute(response)
            self.assertFalse(row["valid"])
            self.assertNotIn(self.key, json.dumps(row))

    def test_dry_run_has_no_private_configuration_reads_network_or_output(self):
        args = api.arguments(["--bundle", str(self.bundle), "--dry-run"])
        with patch.object(api, "load_config") as config, patch.object(api.httpx, "Client") as client, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(api.run(args), 0)
        config.assert_not_called()
        client.assert_not_called()
        self.assertFalse((self.bundle / "results").exists())

    def test_every_referenced_input_must_be_covered_by_manifest(self):
        manifest_path = self.bundle / "manifest.json"
        for name in ("system_prompt.txt", self.case["entry"]["input_file"], *self.case["model"]["images"]):
            with self.subTest(asset=name):
                manifest = copy.deepcopy(self.manifest)
                del manifest["files"][name]
                manifest_path.write_text(json.dumps(manifest))
                args = api.arguments(["--bundle", str(self.bundle), "--dry-run"])
                with patch.object(api.httpx, "Client") as client, self.assertRaisesRegex(ValueError, "hash manifest"):
                    api.run(args)
                client.assert_not_called()
                self.assertFalse((self.bundle / "results").exists())

    def test_invalid_hash_manifest_and_unsafe_case_ids_rejected(self):
        for change in (lambda m: m.update(files={}),
                       lambda m: m["files"].update({"system_prompt.txt": "not-a-sha256"}),
                       lambda m: m["cases"][0].update(case_id="../outside")):
            manifest = copy.deepcopy(self.manifest)
            change(manifest)
            (self.bundle / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                api.common.load_bundle(self.bundle)

    def test_three_images_cannot_alias_the_same_asset(self):
        path = self.bundle / self.case["entry"]["input_file"]
        model = api.common.read_json(path)
        model["images"][1] = model["images"][0]
        path.write_text(json.dumps(model))
        self.manifest["files"][self.case["entry"]["input_file"]] = pack.sha(path)
        (self.bundle / "manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "distinct image"):
            api.common.load_bundle(self.bundle)

    def test_public_defaults_are_effective_and_cli_overrides_take_priority(self):
        config = api.common.read_json(api.HERE / "configs/project.json")
        config["inference"].update(rate_hz=.25, period_s=4, deadline_s=3, max_tokens=512)
        config["benchmark"].update(limit=7, timeout_s=45)
        config["development"]["bundle"] = "data/datasets/alternate_fixture"
        with patch.object(api.common, "read_json", return_value=config):
            args = api.arguments([])
            self.assertEqual((args.rate_hz, args.deadline_s, args.max_tokens, args.limit, args.timeout),
                             (.25, 3, 512, 7, 45))
            self.assertEqual(args.bundle, api.HERE.parents[1] / "data/datasets/alternate_fixture")
            args = api.arguments(["--rate-hz", "1", "--limit", "2"])
            self.assertEqual((args.rate_hz, args.deadline_s, args.limit), (1, 1, 2))
            args = api.arguments(["--rate-hz", "1", "--deadline-s", "1.5"])
            self.assertEqual((args.rate_hz, args.deadline_s), (1, 1.5))

    def test_inconsistent_or_unsupported_public_defaults_fail_closed(self):
        original = api.common.read_json(api.HERE / "configs/project.json")
        for section, field, value in (("inference", "period_s", 4), ("inference", "thinking", True),
                                      ("benchmark", "warmups", 2), ("inference", "max_tokens", True)):
            config = copy.deepcopy(original)
            config[section][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                defaults.from_project(config)

    def test_invalid_cli_timing_rejected_before_output_or_network(self):
        for flags in (["--rate-hz", "0"], ["--deadline-s", "nan"], ["--timeout", "301"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                api.arguments(flags)
            self.assertEqual(error.exception.code, 2)

    def test_default_run_one_get_three_unique_image_tasks_no_retries(self):
        requests, bodies = [], []
        def handler(request):
            requests.append(request)
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": MODEL}]})
            bodies.append(json.loads(request.content))
            return stream()
        result, factory = self.run_mock(handler)
        self.assertEqual(result, 0)
        self.assertEqual([r.method for r in requests], ["GET", "POST", "POST", "POST"])
        self.assertEqual(len({b["messages"][1]["content"][-1]["text"] for b in bodies}), 3)
        self.assertFalse(factory.call_args.kwargs["trust_env"])
        self.assertFalse(factory.call_args.kwargs["follow_redirects"])
        report = api.common.read_json(self.root / "results/summary.json")
        self.assertEqual((report["valid"], report["failed"]), (3, 0))
        self.assertNotIn(self.key, (self.root / "results/settings.json").read_text())

    def test_check_mode_has_no_inference(self):
        seen = []
        def handler(request):
            seen.append(request.method)
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        result, _ = self.run_mock(handler, ["--check"])
        self.assertEqual(result, 0)
        self.assertEqual(seen, ["GET"])
        self.assertFalse((self.root / "results/requests.jsonl").exists())

    def test_first_failed_task_stops_without_fallback_or_automatic_retry(self):
        seen = []
        def handler(request):
            seen.append(request.method)
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": MODEL}]})
            return httpx.Response(400, text=self.key)
        result, _ = self.run_mock(handler)
        self.assertEqual(result, 1)
        self.assertEqual(seen, ["GET", "POST"])
        report = api.common.read_json(self.root / "results/summary.json")
        self.assertEqual((report["submitted"], report["failed"], report["cancelled_after_error"]), (1, 1, 2))
        self.assertNotIn(self.key, (self.root / "results/requests.jsonl").read_text())

    def test_timeout_stops_without_retry_and_does_not_log_exception_message(self):
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": MODEL}]})
            raise httpx.ReadTimeout(self.key, request=request)
        result, _ = self.run_mock(handler)
        self.assertEqual(result, 1)
        log = (self.root / "results/requests.jsonl").read_text()
        self.assertEqual(len(log.splitlines()), 1)
        self.assertNotIn(self.key, log)
        self.assertIn("ReadTimeout", log)

    def test_prefix_warmup_is_explicit_single_text_request(self):
        bodies = []
        def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": MODEL}]})
            body = json.loads(request.content)
            bodies.append(body)
            return stream('{"g":[],"u":[]}' if len(bodies) == 1 else '{"g":[{"m":[0,1],"r":"w","k":0}],"u":[]}')
        result, _ = self.run_mock(handler, ["--warmups", "1"])
        self.assertEqual(result, 0)
        self.assertEqual(len(bodies), 4)
        self.assertNotIn("image_url", json.dumps(bodies[0]))
        self.assertFalse(bodies[0]["chat_template_kwargs"]["enable_thinking"])

    def test_busy_paced_arrivals_are_dropped_not_queued(self):
        def handler(request):
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        def execute(*args):
            target = args[-1]
            return {"case_id": "case_00", "submitted": True, "valid": True, "status": "success",
                    "valid_within_deadline": False, "timing": {"end_to_end_s": 2.1}}, target + 2.1
        result, _ = self.run_mock(handler, ["--mode", "paced", "--rate-hz", "1"], execute=execute)
        self.assertEqual(result, 0)
        report = api.common.read_json(self.root / "results/summary.json")
        self.assertEqual((report["submitted"], report["dropped_busy"], report["planned_arrivals"]), (1, 2, 3))
        self.assertEqual(report["valid_within_deadline_ratio_all_arrivals"], 0)

    def test_existing_output_is_preserved(self):
        out = self.root / "results"
        out.mkdir()
        (out / "user.txt").write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.run_mock(lambda request: self.fail("No network allowed"))
        self.assertEqual((out / "user.txt").read_text(), "preserve")

    def test_offline_packager_only_copies_whitelisted_inputs_with_blank_config(self):
        source = self.root / "source24"
        manifest, _, _ = fixture(source, 24)
        # Sensitive-looking source files must never be read or recursively copied.
        (source / "api_key.json").write_text("SYNTHETIC_SECRET_DO_NOT_COPY")
        (source / "native_labels.json").write_text("SYNTHETIC_GOLD_DO_NOT_COPY")
        out = self.root / "portable"
        result = pack.package(out, source)
        self.assertEqual((result["cases"], result["images"], result["inference_requests_sent"]), (24, 72, 0))
        config = api.common.read_json(out / "server_config.json")
        self.assertEqual(config, {"base_url": "", "api_key": "", "model": "", "allow_insecure_http": False})
        packed_manifest, _, _ = api.common.load_bundle(out)
        self.assertNotIn('version', packed_manifest)
        self.assertEqual(packed_manifest['prompt_profile'], 'object')
        self.assertEqual(len(packed_manifest["source_frozen_input_files"]), 97)
        self.assertNotIn("server_config.json", packed_manifest["files"])
        self.assertEqual(packed_manifest["source_frozen_input_files"], manifest["files"])
        exported = api.common.read_json(out / "benchmark_defaults.json")
        self.assertEqual(set(exported), {'settings'})
        self.assertEqual(exported, packed_manifest["defaults"])
        self.assertEqual(defaults.load_portable(out), defaults.from_project(
            api.common.read_json(api.HERE / "configs/project.json")))
        with zipfile.ZipFile(result["archive"]) as archive:
            self.assertIsNone(archive.testzip())
            names = {n.split("/", 1)[1] for n in archive.namelist()}
            self.assertEqual(names, set(packed_manifest["files"]) | {"manifest.json", "server_config.json"})
            self.assertNotIn("api_key.json", names)
            self.assertNotIn("native_labels.json", names)
            self.assertIn("README.md", names)
            self.assertNotIn("README.txt", names)
            self.assertNotIn("benchmark_guide.md", names)
        self.assertEqual((out / 'README.md').read_bytes(),
                         (api.HERE / 'configs/vlm_test_guide.md').read_bytes())
        with self.assertRaises(ValueError):
            pack.package(out, source)

    def test_corrupt_source_is_rejected_before_output_created(self):
        source = self.root / "corrupt24"
        _, _, cases = fixture(source, 24)
        cases[0]["paths"][0].write_bytes(b"CORRUPTED")
        out = self.root / "not_created"
        with self.assertRaises(ValueError):
            pack.package(out, source)
        self.assertFalse(out.exists())

    def test_deployment_package_reuses_client_assets_and_contains_no_existing_credentials(self):
        out = self.root / 'deployment'
        result = pack.package(out, self.bundle, include_deployment=True)
        manifest, _, cases = api.common.load_bundle(out)
        self.assertTrue(result['deployment_guide_included'])
        self.assertFalse(manifest['deployment']['remote_installation_or_gpu_inference_verified'])
        self.assertEqual(manifest['deployment']['installation_mode'], 'current_environment')
        self.assertTrue(manifest['deployment']['environment_will_be_modified_on_install'])
        self.assertFalse(manifest['deployment']['base_environment_modified_during_packaging'])
        self.assertFalse(manifest['deployment']['creates_virtual_environment'])
        self.assertTrue(manifest['deployment']['offline_wheelhouse_install_supported'])
        self.assertFalse(manifest['deployment']['wheelhouse_included'])
        self.assertEqual(manifest['deployment']['runtime_packages']['cuda-tile'], '1.6.0rc5')
        requirements = (out / 'requirements.server.txt').read_text()
        self.assertIn('--extra-index-url https://pypi.nvidia.com', requirements)
        self.assertIn('--extra-index-url https://download.pytorch.org/whl/cu130', requirements)
        self.assertIn('cuda-tile==1.6.0rc5', requirements)
        config = api.common.read_json(out / 'server_config.json')
        self.assertEqual(config['api_key'], '')
        self.assertEqual(config['base_url'], 'http://127.0.0.1:30000/v1')
        self.assertFalse(config['allow_insecure_http'])
        self.assertEqual(len(cases), 3)
        self.assertEqual((out / 'benchmark.py').read_bytes(), (api.HERE / 'benchmark_vlm.py').read_bytes())
        for name in ('deploy.py', 'README.md', 'benchmark_guide.md', 'installation_profile.json', 'deployment_defaults.json',
                     'requirements.server.txt', 'compare_results.py'):
            self.assertIn(name, manifest['files'])
        self.assertNotIn('README.txt', manifest['files'])
        self.assertEqual((out / 'README.md').read_bytes(), (api.HERE / 'deployment/README.md').read_bytes())
        self.assertEqual((out / 'benchmark_guide.md').read_bytes(),
                         (api.HERE / 'configs/vlm_test_guide.md').read_bytes())
        from decision.vlm.deployment import deploy
        deploy.verify_templates(out)
        (out / 'server_config.json').write_text('INVALID_PRIVATE_CONFIG_MUST_NOT_BE_READ')
        deploy.verify_templates(out)
        (out / 'installation_profile.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed'):
            deploy.verify_templates(out)

    def test_portable_defaults_tampering_and_missing_hash_rejected(self):
        out = self.root / "portable"
        pack.package(out, self.bundle)
        manifest_path = out / "manifest.json"
        manifest = api.common.read_json(manifest_path)
        del manifest["files"]["benchmark_defaults.json"]
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            defaults.load_portable(out)
        manifest["files"]["benchmark_defaults.json"] = pack.sha(out / "benchmark_defaults.json")
        manifest_path.write_text(json.dumps(manifest))
        settings = api.common.read_json(out / "benchmark_defaults.json")
        settings["settings"]["max_tokens"] = 256
        (out / "benchmark_defaults.json").write_text(json.dumps(settings))
        with self.assertRaises(ValueError):
            defaults.load_portable(out)

    def test_package_cannot_write_inside_frozen_source(self):
        out = self.bundle / "new_package"
        with self.assertRaisesRegex(ValueError, "outside the source"):
            pack.package(out, self.bundle)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
