import importlib.util
import os
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "qwen_tts" / "inference" / "qwen3_tts_model.py"
TOKENIZER_PATH = REPO_ROOT / "qwen_tts" / "inference" / "qwen3_tts_tokenizer.py"


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


def build_import_stubs():
    # Lightweight stubs to import modules without runtime ML dependencies.
    np_mod = types.ModuleType("numpy")
    np_mod.ndarray = type("ndarray", (), {})
    np_mod.float32 = "float32"
    np_mod.asarray = lambda x, dtype=None: x
    np_mod.mean = lambda x, axis=-1: x

    librosa_mod = types.ModuleType("librosa")
    librosa_mod.load = lambda *args, **kwargs: ([], 16000)
    librosa_mod.resample = lambda y, orig_sr, target_sr: y

    sf_mod = types.ModuleType("soundfile")
    sf_mod.read = lambda *args, **kwargs: ([], 16000)

    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = object
    torch_mod.device = lambda x=None: x

    def _identity_decorator(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        return lambda fn: fn

    torch_mod.inference_mode = _identity_decorator
    torch_mod.no_grad = _identity_decorator
    nn_mod = types.ModuleType("torch.nn")
    nn_utils_mod = types.ModuleType("torch.nn.utils")
    nn_utils_rnn_mod = types.ModuleType("torch.nn.utils.rnn")
    nn_utils_rnn_mod.pad_sequence = lambda *args, **kwargs: None
    nn_utils_mod.rnn = nn_utils_rnn_mod
    nn_mod.utils = nn_utils_mod
    torch_mod.nn = nn_mod

    transformers_mod = types.ModuleType("transformers")

    class _DummyAuto:
        @classmethod
        def register(cls, *args, **kwargs):
            return None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return object()

    transformers_mod.AutoConfig = _DummyAuto
    transformers_mod.AutoModel = _DummyAuto
    transformers_mod.AutoProcessor = _DummyAuto
    transformers_mod.AutoFeatureExtractor = _DummyAuto

    pkg_qwen = types.ModuleType("qwen_tts")
    pkg_qwen.__path__ = []
    pkg_inference = types.ModuleType("qwen_tts.inference")
    pkg_inference.__path__ = []
    pkg_core = types.ModuleType("qwen_tts.core")
    pkg_core.__path__ = []
    pkg_models = types.ModuleType("qwen_tts.core.models")

    class _Dummy:
        pass

    pkg_models.Qwen3TTSConfig = _Dummy
    pkg_models.Qwen3TTSForConditionalGeneration = _Dummy
    pkg_models.Qwen3TTSProcessor = _Dummy

    pkg_core.Qwen3TTSTokenizerV1Config = _Dummy
    pkg_core.Qwen3TTSTokenizerV1Model = _Dummy
    pkg_core.Qwen3TTSTokenizerV2Config = _Dummy
    pkg_core.Qwen3TTSTokenizerV2Model = _Dummy

    return {
        "numpy": np_mod,
        "librosa": librosa_mod,
        "soundfile": sf_mod,
        "torch": torch_mod,
        "torch.nn": nn_mod,
        "torch.nn.utils": nn_utils_mod,
        "torch.nn.utils.rnn": nn_utils_rnn_mod,
        "transformers": transformers_mod,
        "qwen_tts": pkg_qwen,
        "qwen_tts.inference": pkg_inference,
        "qwen_tts.core": pkg_core,
        "qwen_tts.core.models": pkg_models,
    }


def load_module(module_name: str, path: Path):
    overrides = build_import_stubs()
    with temporary_modules(overrides):
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            assert spec.loader is not None
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
        return module


class FakeResponse:
    def __init__(self, chunks, headers=None):
        self._chunks = list(chunks)
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, _size=-1):
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class TestRemoteAudioSecurityGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_module = load_module(
            "qwen_tts.inference.qwen3_tts_model",
            MODEL_PATH,
        )
        cls.tokenizer_module = load_module(
            "qwen_tts.inference.qwen3_tts_tokenizer",
            TOKENIZER_PATH,
        )
        cls.targets = [
            (cls.model_module, cls.model_module.Qwen3TTSModel),
            (cls.tokenizer_module, cls.tokenizer_module.Qwen3TTSTokenizer),
        ]

    def test_is_private_host_classification(self):
        for module, klass in self.targets:
            inst = klass.__new__(klass)
            with self.subTest(target=klass.__name__, case="private_ip"):
                with mock.patch.object(
                    module.socket,
                    "getaddrinfo",
                    return_value=[(None, None, None, None, ("10.0.0.1", 0))],
                ):
                    self.assertTrue(inst._is_private_host("example.local"))

            with self.subTest(target=klass.__name__, case="public_ip"):
                with mock.patch.object(
                    module.socket,
                    "getaddrinfo",
                    return_value=[(None, None, None, None, ("8.8.8.8", 0))],
                ):
                    self.assertFalse(inst._is_private_host("example.com"))

            with self.subTest(target=klass.__name__, case="resolution_error"):
                with mock.patch.object(
                    module.socket,
                    "getaddrinfo",
                    side_effect=OSError("dns failed"),
                ):
                    self.assertTrue(inst._is_private_host("bad.host"))

    def test_read_remote_audio_blocks_private_host_by_default(self):
        for _module, klass in self.targets:
            inst = klass.__new__(klass)
            with self.subTest(target=klass.__name__):
                with mock.patch.object(inst, "_is_private_host", return_value=True):
                    with mock.patch.dict(
                        os.environ,
                        {"QWEN_TTS_ALLOW_PRIVATE_URLS": ""},
                        clear=False,
                    ):
                        with self.assertRaisesRegex(ValueError, "Refusing to fetch audio"):
                            inst._read_remote_audio_bytes("http://localhost/a.wav", max_bytes=8)

    def test_read_remote_audio_allows_private_host_with_override(self):
        for module, klass in self.targets:
            inst = klass.__new__(klass)
            with self.subTest(target=klass.__name__):
                with mock.patch.object(inst, "_is_private_host", return_value=True):
                    with mock.patch.dict(
                        os.environ,
                        {"QWEN_TTS_ALLOW_PRIVATE_URLS": "1"},
                        clear=False,
                    ):
                        with mock.patch.object(
                            module.urllib.request,
                            "urlopen",
                            return_value=FakeResponse([b"abc", b""]),
                        ):
                            out = inst._read_remote_audio_bytes(
                                "http://localhost/a.wav",
                                max_bytes=8,
                            )
                self.assertEqual(out, b"abc")

    def test_read_remote_audio_rejects_oversized_content_length(self):
        for module, klass in self.targets:
            inst = klass.__new__(klass)
            with self.subTest(target=klass.__name__):
                with mock.patch.object(inst, "_is_private_host", return_value=False):
                    with mock.patch.object(
                        module.urllib.request,
                        "urlopen",
                        return_value=FakeResponse(
                            [b""],
                            headers={"Content-Length": "100"},
                        ),
                    ):
                        with self.assertRaisesRegex(ValueError, "Remote audio too large"):
                            inst._read_remote_audio_bytes(
                                "https://example.com/a.wav",
                                max_bytes=16,
                            )

    def test_read_remote_audio_rejects_stream_over_limit(self):
        for module, klass in self.targets:
            inst = klass.__new__(klass)
            with self.subTest(target=klass.__name__):
                with mock.patch.object(inst, "_is_private_host", return_value=False):
                    with mock.patch.object(
                        module.urllib.request,
                        "urlopen",
                        return_value=FakeResponse([b"abcd", b"ef", b""]),
                    ):
                        with self.assertRaisesRegex(
                            ValueError,
                            "Remote audio exceeded maximum allowed size",
                        ):
                            inst._read_remote_audio_bytes(
                                "https://example.com/a.wav",
                                max_bytes=5,
                            )


if __name__ == "__main__":
    unittest.main()
