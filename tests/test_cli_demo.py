import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = REPO_ROOT / "qwen_tts" / "cli" / "demo.py"


@contextmanager
def temporary_modules(overrides):
    sentinel = object()
    original = {}
    for name, module in overrides.items():
        original[name] = sys.modules.get(name, sentinel)
        sys.modules[name] = module
    try:
        yield
    finally:
        for name, value in original.items():
            if value is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


@contextmanager
def loaded_demo_module():
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = object()
    fake_torch.float16 = object()
    fake_torch.float32 = object()
    fake_torch.dtype = object

    fake_pkg = types.ModuleType("qwen_tts")
    fake_pkg.__path__ = []
    fake_pkg.Qwen3TTSModel = type("DummyQwen3TTSModel", (), {})

    fake_cli = types.ModuleType("qwen_tts.cli")
    fake_cli.__path__ = []

    fake_ui = types.ModuleType("qwen_tts.ui")
    fake_ui.build_demo = lambda *_args, **_kwargs: None

    with temporary_modules(
        {
            "torch": fake_torch,
            "qwen_tts": fake_pkg,
            "qwen_tts.cli": fake_cli,
            "qwen_tts.ui": fake_ui,
        }
    ):
        spec = importlib.util.spec_from_file_location("qwen_tts.cli.demo", DEMO_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules["qwen_tts.cli.demo"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        try:
            yield module, fake_torch
        finally:
            sys.modules.pop("qwen_tts.cli.demo", None)


class TestCliDemo(unittest.TestCase):
    def test_dtype_aliases_and_error(self):
        with loaded_demo_module() as (demo, fake_torch):
            self.assertIs(demo._dtype_from_str("bf16"), fake_torch.bfloat16)
            self.assertIs(demo._dtype_from_str("BFloat16"), fake_torch.bfloat16)
            self.assertIs(demo._dtype_from_str("fp16"), fake_torch.float16)
            self.assertIs(demo._dtype_from_str("float32"), fake_torch.float32)
            with self.assertRaises(ValueError):
                demo._dtype_from_str("int8")

    def test_collect_gen_kwargs_filters_none(self):
        with loaded_demo_module() as (demo, _fake_torch):
            args = demo.build_parser().parse_args(
                [
                    "dummy/checkpoint",
                    "--max-new-tokens",
                    "64",
                    "--temperature",
                    "0.7",
                    "--top-k",
                    "30",
                ]
            )
            kwargs = demo._collect_gen_kwargs(args)
            self.assertEqual(
                kwargs,
                {
                    "max_new_tokens": 64,
                    "temperature": 0.7,
                    "top_k": 30,
                },
            )

    def test_main_windows_policy_and_launch_kwargs(self):
        with loaded_demo_module() as (demo, fake_torch):
            from_pretrained_calls = []
            build_demo_calls = []
            queue_calls = []
            launch_calls = []
            policy_calls = []

            class FakeModel:
                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    from_pretrained_calls.append((args, kwargs))
                    return "FAKE_TTS"

            class FakeDemo:
                _launch_extras = {"theme": "test-theme"}

                def queue(self, default_concurrency_limit):
                    queue_calls.append(default_concurrency_limit)
                    return self

                def launch(self, **kwargs):
                    launch_calls.append(kwargs)
                    return None

            def fake_build_demo(tts, ckpt, gen_kwargs_default, attn_impl):
                build_demo_calls.append((tts, ckpt, gen_kwargs_default, attn_impl))
                return FakeDemo()

            class FakePolicy:
                pass

            policy = FakePolicy()
            demo.sys.platform = "win32"
            demo.asyncio.WindowsSelectorEventLoopPolicy = lambda: policy
            demo.asyncio.set_event_loop_policy = lambda p: policy_calls.append(p)
            demo.Qwen3TTSModel = FakeModel
            demo.build_demo = fake_build_demo

            rc = demo.main(
                [
                    "Qwen/mock-checkpoint",
                    "--dtype",
                    "fp16",
                    "--no-flash-attn",
                    "--ip",
                    "127.0.0.1",
                    "--port",
                    "19090",
                    "--share",
                    "--concurrency",
                    "3",
                    "--no-ssl-verify",
                    "--max-new-tokens",
                    "77",
                ]
            )

            self.assertEqual(rc, 0)
            self.assertEqual(policy_calls, [policy])
            self.assertEqual(queue_calls, [3])
            self.assertEqual(len(from_pretrained_calls), 1)
            self.assertEqual(len(build_demo_calls), 1)
            self.assertEqual(len(launch_calls), 1)

            args, kwargs = from_pretrained_calls[0]
            self.assertEqual(args, ("Qwen/mock-checkpoint",))
            self.assertEqual(kwargs["device_map"], "cuda:0")
            self.assertIs(kwargs["dtype"], fake_torch.float16)
            self.assertIsNone(kwargs["attn_implementation"])

            demo_args = build_demo_calls[0]
            self.assertEqual(demo_args[0], "FAKE_TTS")
            self.assertEqual(demo_args[1], "Qwen/mock-checkpoint")
            self.assertEqual(demo_args[2], {"max_new_tokens": 77})
            self.assertIsNone(demo_args[3])

            launch_kwargs = launch_calls[0]
            self.assertEqual(launch_kwargs["server_name"], "127.0.0.1")
            self.assertEqual(launch_kwargs["server_port"], 19090)
            self.assertTrue(launch_kwargs["share"])
            self.assertFalse(launch_kwargs["ssl_verify"])
            self.assertEqual(launch_kwargs["theme"], "test-theme")


if __name__ == "__main__":
    unittest.main()
