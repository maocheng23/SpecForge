"""Capture snapshots must survive scheduler reuse and writer-thread handoff."""

from __future__ import annotations

import ctypes
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from typing import Optional
from unittest import mock

import torch

PATCH = (
    Path(__file__).resolve().parents[2] / "patches/sglang/v0.5.18/spec-capture.patch"
)


def _added_source(path):
    section = next(
        part
        for part in PATCH.read_text().split("diff --git ")
        if part.startswith(f"a/{path} b/{path}\n")
    )
    return "\n".join(
        line[1:]
        for line in section.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def _load_sink():
    module = types.ModuleType("sglang.srt.spec_capture_sink")
    exec(_added_source("python/sglang/srt/spec_capture_sink.py"), module.__dict__)
    return module


def _append_capture():
    source = _added_source(
        "python/sglang/srt/managers/scheduler_components/batch_result_processor.py"
    )
    start = source.index("    def _append_spec_capture_states(")
    end = source.index("    def _sink_spec_capture(", start)
    namespace = {}
    exec(
        "from __future__ import annotations\n" + textwrap.dedent(source[start:end]),
        namespace,
    )
    return namespace["_append_spec_capture_states"]


class _BufferStore:
    def __init__(self, *, device=False, register_status=0):
        self.device = device
        self.register_status = register_status
        self.registrations = []
        self.unregistered = []
        self.values = {}

    def register_buffer(self, ptr, size):
        self.registrations.append((ptr, size))
        return self.register_status

    def unregister_buffer(self, ptr):
        self.unregistered.append(ptr)
        return 0

    def batch_put_from(self, keys, pointers, sizes, config):
        for key, ptr, size in zip(keys, pointers, sizes):
            if self.device:
                pointer = types.SimpleNamespace(
                    __cuda_array_interface__={
                        "shape": (size,),
                        "typestr": "|u1",
                        "data": (ptr, False),
                        "version": 3,
                    }
                )
                self.values[key] = (
                    torch.as_tensor(pointer, device="cuda").cpu().numpy().tobytes()
                )
            else:
                self.values[key] = ctypes.string_at(ptr, size)
        return [0] * len(keys)


class SpecCaptureSinkTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_sink()
        self.sink = self.module.SpecCaptureSink()
        self.addCleanup(self.sink._executor.shutdown)
        self.env = mock.patch.dict(
            "os.environ",
            {"SGLANG_SPEC_CAPTURE_GPU_PUT": "0", "MOONCAKE_PROTOCOL": "rdma"},
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def _request(self, sample_id):
        return types.SimpleNamespace(
            spec_capture={
                "store_id": "test",
                "sample_id": sample_id,
                "features": {"aux": "hidden_states"},
            },
            spec_capture_aux=[],
            spec_capture_last_hidden=[],
        )

    def _append(self, request, logits, start, length):
        processor = types.SimpleNamespace(
            output_streamer=types.SimpleNamespace(
                ps=types.SimpleNamespace(attn_tp_rank=0)
            )
        )
        with mock.patch.dict(
            sys.modules, {"sglang.srt.spec_capture_sink": self.module}
        ):
            return _append_capture()(
                processor,
                req=request,
                logits_output=logits,
                hidden_state_offset=start,
                extend_input_len=length,
            )

    def test_batch_views_register_their_shared_storage_once(self):
        backing = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        requests = [self._request("a"), self._request("b")]
        logits = types.SimpleNamespace(hidden_states=backing, last_hidden_states=None)
        self._append(requests[0], logits, 0, 2)
        self._append(requests[1], logits, 2, 4)
        store = self.sink._store = _BufferStore()
        self.sink.submit_samples(
            [(req.spec_capture, req.spec_capture_aux[0], None) for req in requests]
        ).result(timeout=5)
        self.assertEqual(
            store.registrations, [(backing.data_ptr(), backing.numel() * 4)]
        )
        self.assertEqual(store.unregistered, [backing.data_ptr()])
        self.assertEqual(
            store.values["test/a/g1/hidden_states"], backing[:2].numpy().tobytes()
        )
        self.assertEqual(
            store.values["test/b/g1/hidden_states"], backing[2:].numpy().tobytes()
        )

    def test_gpu_capture_rejects_premature_host_staging(self):
        with mock.patch.dict("os.environ", {"SGLANG_SPEC_CAPTURE_GPU_PUT": "1"}):
            with self.assertRaisesRegex(RuntimeError, "CPU staging"):
                self._append(
                    self._request("a"),
                    types.SimpleNamespace(hidden_states=torch.ones(2, 4)),
                    0,
                    2,
                )

    def test_external_gpu_put_rejects_tcp(self):
        with mock.patch.dict(
            "os.environ",
            {"SGLANG_SPEC_CAPTURE_GPU_PUT": "1", "MOONCAKE_PROTOCOL": "tcp"},
        ):
            with self.assertRaisesRegex(ValueError, "requires MOONCAKE_PROTOCOL=rdma"):
                self.module.gpu_put_enabled()

    def test_default_publication_requires_cuda_and_rdma(self):
        for protocol in ("tcp", "rdma"):
            for cuda_version in (None, "13.0"):
                for available in (False, True):
                    with self.subTest(
                        protocol=protocol, cuda=cuda_version, available=available
                    ):
                        with (
                            mock.patch.dict(
                                "os.environ",
                                {"MOONCAKE_PROTOCOL": protocol},
                                clear=True,
                            ),
                            mock.patch.object(torch.version, "cuda", cuda_version),
                            mock.patch.object(
                                torch.cuda, "is_available", return_value=available
                            ),
                        ):
                            self.assertEqual(
                                self.module.gpu_put_enabled(),
                                protocol == "rdma"
                                and cuda_version is not None
                                and available,
                            )

    def test_explicit_host_publication_overrides_the_rdma_default(self):
        with (
            mock.patch.object(torch.version, "cuda", "13.0"),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
        ):
            self.assertFalse(self.module.gpu_put_enabled())

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_gpu_snapshot_survives_overwrite_and_stream_handoff(self):
        store = self.sink._store = _BufferStore(device=True)
        request = self._request("gpu")
        stream = torch.cuda.Stream()
        with mock.patch.dict("os.environ", {"SGLANG_SPEC_CAPTURE_GPU_PUT": "1"}):
            with torch.cuda.stream(stream):
                torch.cuda._sleep(10_000_000)
                source = torch.full((4096, 8), 7.0, device="cuda")
                logits = types.SimpleNamespace(hidden_states=source)
                self._append(request, logits, 0, source.shape[0])
                source.zero_()
                future = self.sink.submit_samples(
                    [(request.spec_capture, request.spec_capture_aux[0], None)]
                )
            future.result(timeout=20)
        expected = torch.full((4096, 8), 7.0).numpy().tobytes()
        self.assertEqual(store.values["test/gpu/g1/hidden_states"], expected)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_snapshot_finishes_before_forward_can_overwrite_graph_storage(self):
        source = _added_source("python/sglang/srt/managers/utils.py")
        start = source.index("        if capture_stream is not None:")
        last = "            capture_stream.wait_stream(torch.cuda.current_stream())"
        end = source.index(last, start) + len(last)
        namespace = {"torch": torch}
        exec("def snapshot(self, capture_stream):\n" + source[start:end], namespace)
        forward = torch.cuda.Stream()
        copier = torch.cuda.Stream()
        with torch.cuda.stream(forward):
            backing = torch.full((4096, 8), 7.0, device="cuda")
        logits = types.SimpleNamespace(hidden_states=backing, last_hidden_states=None)
        result = types.SimpleNamespace(logits_output=logits)
        copier.wait_stream(forward)
        with torch.cuda.stream(copier):
            torch.cuda._sleep(10_000_000)
            namespace["snapshot"](result, forward)
        with torch.cuda.stream(forward):
            backing.zero_()
        forward.synchronize()
        self.assertTrue(torch.all(logits.spec_capture_aux_gpu == 7).item())

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_gpu_registration_failure_prevents_publication(self):
        store = self.sink._store = _BufferStore(device=True, register_status=-600)
        with mock.patch.dict("os.environ", {"SGLANG_SPEC_CAPTURE_GPU_PUT": "1"}):
            with self.assertRaisesRegex(RuntimeError, "registration failed"):
                self.sink.put_samples(
                    [
                        (
                            self._request("gpu").spec_capture,
                            torch.ones(2, 4, device="cuda"),
                            None,
                        )
                    ]
                )
        self.assertEqual(store.values, {})
        self.assertEqual(store.unregistered, [])


def _skip_tail_layer_method():
    source = _added_source("python/sglang/srt/models/qwen3_5.py")
    start = source.index("    def _spec_capture_skip_tail_layer(")
    end = source.index("        return max(tap_layers)\n", start) + len(
        "        return max(tap_layers)\n"
    )
    namespace = {"Optional": Optional}
    exec(
        "from __future__ import annotations\n" + textwrap.dedent(source[start:end]),
        namespace,
    )
    return namespace


class CaptureFastPathGateTest(unittest.TestCase):
    """The experimental capture-server gates stay off and fail closed."""

    def setUp(self):
        self.module = _load_sink()
        self.env = mock.patch.dict(
            "os.environ",
            {
                name: ""
                for name in (
                    "SGLANG_SPEC_CAPTURE_SKIP_TAIL",
                    "SGLANG_CAPTURE_FA_SKIP_KV",
                    "SGLANG_SPEC_CAPTURE_DEBUG_DUMP",
                )
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        parallel = mock.patch.object(
            self.module, "_parallel_layout_blocks_fast_paths", return_value=False
        )
        parallel.start()
        self.addCleanup(parallel.stop)

    @staticmethod
    def _req(features, *, capture=True):
        return types.SimpleNamespace(
            spec_capture=(
                {"store_id": "s", "sample_id": "x", "features": features}
                if capture
                else None
            )
        )

    @staticmethod
    def _batch(reqs, **overrides):
        fields = dict(
            reqs=reqs,
            is_prefill_only=True,
            return_logprob=False,
            forward_mode=types.SimpleNamespace(name="EXTEND"),
            global_num_tokens=None,
            tbo_split_seq_index=None,
            prefix_lens=[0] * len(reqs),
        )
        fields.update(overrides)
        return types.SimpleNamespace(**fields)

    def test_aux_only_capture_batch_qualifies_for_both_paths(self):
        batch = self._batch([self._req({"aux": "hidden_states"})] * 3)
        self.assertEqual(self.module.capture_prefill_flags(batch), (True, True))

    def test_last_hidden_request_keeps_the_full_model(self):
        batch = self._batch(
            [
                self._req({"aux": "hidden_states"}),
                self._req({"aux": "hidden_states", "last_hidden": "target"}),
            ]
        )
        self.assertEqual(self.module.capture_prefill_flags(batch), (True, False))

    def test_any_non_capture_or_decoding_batch_stays_stock(self):
        aux = self._req({"aux": "hidden_states"})
        cases = {
            "non-capture request": self._batch([aux, self._req({}, capture=False)]),
            "decoding request": self._batch([aux], is_prefill_only=False),
            "logprobs": self._batch([aux], return_logprob=True),
            "mixed mode": self._batch(
                [aux], forward_mode=types.SimpleNamespace(name="MIXED")
            ),
            "cached prefix": self._batch([aux], prefix_lens=[16]),
            "dp attention sync": self._batch([aux], global_num_tokens=[4, 4]),
            "two-batch overlap": self._batch([aux], tbo_split_seq_index=1),
            "empty": self._batch([]),
        }
        for name, batch in cases.items():
            with self.subTest(name):
                self.assertEqual(
                    self.module.capture_prefill_flags(batch), (False, False)
                )
        with mock.patch.object(
            self.module, "_parallel_layout_blocks_fast_paths", return_value=True
        ):
            self.assertEqual(
                self.module.capture_prefill_flags(self._batch([aux])), (False, False)
            )

    def test_annotation_is_inert_until_a_gate_is_set(self):
        batch = self._batch([self._req({"aux": "hidden_states"})])
        teacher = self._batch(
            [self._req({"aux": "hidden_states", "last_hidden": "target"})]
        )
        no_radix = types.SimpleNamespace(disable_radix_cache=True)
        radix = types.SimpleNamespace(disable_radix_cache=False)
        skip_tail = {"SGLANG_SPEC_CAPTURE_SKIP_TAIL": "1"}
        cases = (
            ({}, batch, no_radix, (None, None), None),
            ({"SGLANG_CAPTURE_FA_SKIP_KV": "1"}, batch, no_radix, (True, None), None),
            (skip_tail, batch, no_radix, (True, True), None),
            # Skipped layers leave KV/recurrent state unwritten: never cache it.
            (skip_tail, batch, radix, (True, None), "the radix cache is on"),
            (skip_tail, teacher, no_radix, (True, None), "asks for last_hidden"),
        )
        for env, schedule_batch, server_args, expected, ignored in cases:
            with self.subTest(env=env, server_args=server_args, ignored=ignored):
                with (
                    mock.patch.dict("os.environ", env),
                    mock.patch.object(self.module, "log_once") as log_once,
                ):
                    forward_batch = types.SimpleNamespace(
                        spec_capture_prefill=None, spec_capture_skip_tail=None
                    )
                    self.module.annotate_capture_forward_batch(
                        forward_batch, schedule_batch, server_args
                    )
                self.assertEqual(
                    (
                        forward_batch.spec_capture_prefill,
                        forward_batch.spec_capture_skip_tail,
                    ),
                    expected,
                )
                # A skip-tail server that cannot skip says why, once.
                ignored_lines = [
                    call.args[0]
                    for call in log_once.call_args_list
                    if "=1 ignored:" in call.args[0]
                ]
                if ignored is None:
                    self.assertEqual(ignored_lines, [])
                else:
                    (line,) = ignored_lines
                    self.assertTrue(
                        line.startswith("SGLANG_SPEC_CAPTURE_SKIP_TAIL=1 ignored:")
                    )
                    self.assertIn(ignored, line)

    def test_skip_tail_layer_is_the_last_dflash_tap_on_dense_qwen35(self):
        namespace = _skip_tail_layer_method()
        method = namespace["_spec_capture_skip_tail_layer"]
        namespace["get_is_capture_mode"] = lambda: False
        layers = [types.SimpleNamespace() for _ in range(64)]
        for layer_id in (6, 20, 34, 48, 62):
            layers[layer_id]._is_layer_to_capture = True
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(model_type="qwen3_5_text"),
            pp_group=types.SimpleNamespace(is_first_rank=True, is_last_rank=True),
            layers_to_capture=[6, 20, 34, 48, 62],
            layers=layers,
            start_layer=0,
            end_layer=64,
        )
        marked = types.SimpleNamespace(spec_capture_skip_tail=True)

        def ignored_lines(log_once):
            return [
                call.args[0]
                for call in log_once.call_args_list
                if call.args[0].startswith("SGLANG_SPEC_CAPTURE_SKIP_TAIL=1 ignored:")
            ]

        with (
            mock.patch.dict(sys.modules, {"sglang.srt.spec_capture_sink": self.module}),
            mock.patch.object(self.module, "log_once") as log_once,
        ):
            self.assertEqual(method(model, marked), 62)
            self.assertIsNone(
                method(model, types.SimpleNamespace(spec_capture_skip_tail=None))
            )
            namespace["get_is_capture_mode"] = lambda: True
            self.assertIsNone(method(model, marked))
            namespace["get_is_capture_mode"] = lambda: False
            # Unmarked and CUDA-graph-capture batches are not misconfigurations.
            self.assertEqual(ignored_lines(log_once), [])
            # Each fail-closed case of a marked batch logs its reason.
            model.config.model_type = "qwen3_5_moe_text"
            self.assertIsNone(method(model, marked))
            self.assertIn("qwen3_5_moe_text", ignored_lines(log_once)[-1])
            model.config.model_type = "qwen3_5_text"
            model.pp_group.is_last_rank = False
            self.assertIsNone(method(model, marked))
            self.assertIn("pipeline parallelism", ignored_lines(log_once)[-1])
            model.pp_group.is_last_rank = True
            # EAGLE3-style capture lists layers without the DFlash marker.
            del layers[62]._is_layer_to_capture
            self.assertIsNone(method(model, marked))
            self.assertIn("[6, 20, 34, 48, 62]", ignored_lines(log_once)[-1])
            self.assertEqual(len(ignored_lines(log_once)), 3)

    def test_fa_backend_reads_its_gate_before_importing_the_sink(self):
        source = _added_source(
            "python/sglang/srt/layers/attention/flashattention_backend.py"
        )
        gate = f'os.environ.get("{self.module.FA_SKIP_KV_ENV}", "0") == "1"'
        self.assertIn(gate, source)
        self.assertLess(
            source.index(gate),
            source.index("from sglang.srt.spec_capture_sink import"),
        )

    def test_debug_dump_writes_compact_samples_without_mooncake(self):
        sink = self.module.SpecCaptureSink()
        self.addCleanup(sink._executor.shutdown)
        backing = torch.arange(48, dtype=torch.float32).reshape(12, 4)
        spec = {
            "store_id": "store",
            "sample_id": "run:task-1",
            "features": {"aux": "hidden_states"},
            "passthrough": [{"name": "input_ids", "data": [1, 2, 3], "shape": [1, 3]}],
        }
        with (
            tempfile.TemporaryDirectory() as dump_dir,
            mock.patch.dict(
                "os.environ",
                {
                    "SGLANG_SPEC_CAPTURE_DEBUG_DUMP": dump_dir,
                    "SGLANG_SPEC_CAPTURE_GPU_PUT": "0",
                    "MOONCAKE_PROTOCOL": "tcp",
                },
            ),
            mock.patch.object(sink, "_connect", side_effect=AssertionError),
        ):
            (result,) = sink.put_samples([(spec, backing[4:7], None)])
            (path,) = Path(dump_dir).glob("*.pt")
            saved = torch.load(path)
        self.assertEqual(path.name, "store_run_task-1_g1.pt")
        self.assertEqual(saved["key_prefix"], "store/run:task-1/g1")
        aux = saved["features"]["hidden_states"]
        self.assertTrue(torch.equal(aux, backing[4:7].unsqueeze(0)))
        self.assertEqual(aux.untyped_storage().nbytes(), aux.numel() * 4)
        self.assertEqual(saved["features"]["input_ids"].tolist(), [[1, 2, 3]])
        self.assertEqual(result["features"]["hidden_states"]["shape"], [1, 3, 4])

    def test_fa_skip_kv_requires_every_precondition(self):
        backend = types.SimpleNamespace(
            use_mla=False,
            fa_impl_ver=3,
            kv_cache_dtype_str="auto",
            has_local_attention=False,
        )
        server_args = types.SimpleNamespace(
            enable_spec_capture=True,
            chunked_prefill_size=-1,
            disable_radix_cache=True,
        )
        supported = self.module.capture_fa_skip_kv_supported
        self.assertFalse(supported(backend, server_args))
        with mock.patch.dict("os.environ", {"SGLANG_CAPTURE_FA_SKIP_KV": "1"}):
            self.assertTrue(supported(backend, server_args))
            for target, field, value in (
                (server_args, "enable_spec_capture", False),
                (server_args, "chunked_prefill_size", 8192),
                (server_args, "disable_radix_cache", False),
                (backend, "use_mla", True),
                (backend, "fa_impl_ver", 4),
                (backend, "kv_cache_dtype_str", "fp8_e4m3"),
                (backend, "has_local_attention", True),
            ):
                with self.subTest(field=field):
                    original = getattr(target, field)
                    setattr(target, field, value)
                    try:
                        self.assertFalse(supported(backend, server_args))
                    finally:
                        setattr(target, field, original)
