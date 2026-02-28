import builtins
import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEECH_VQ_PATH = REPO_ROOT / "qwen_tts" / "core" / "tokenizer_25hz" / "vq" / "speech_vq.py"


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
def blocked_imports(names):
    blocked = set(names)
    original_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if name in blocked or root in blocked:
            raise ImportError(f"blocked import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = _import
    try:
        yield
    finally:
        builtins.__import__ = original_import


def build_speech_vq_stubs():
    # torch and submodules
    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = object
    torch_mod.log = lambda x: x
    torch_mod.clamp = lambda x, min=None: x
    torch_mod.no_grad = lambda: _NoOpContext()
    torch_mod.from_numpy = lambda x: x
    torch_mod.hann_window = lambda _n: 0
    torch_mod.stft = lambda *args, **kwargs: 0
    torch_mod.view_as_real = lambda x: x
    torch_mod.sqrt = lambda x: x

    nn_mod = types.ModuleType("torch.nn")

    class Module:
        def __init__(self, *args, **kwargs):
            pass

    class Identity(Module):
        pass

    class Conv1d(Module):
        pass

    class ConvTranspose1d(Module):
        pass

    class Linear(Module):
        pass

    nn_mod.Module = Module
    nn_mod.Identity = Identity
    nn_mod.Conv1d = Conv1d
    nn_mod.ConvTranspose1d = ConvTranspose1d
    nn_mod.Linear = Linear

    f_mod = types.ModuleType("torch.nn.functional")
    f_mod.normalize = lambda x, dim=0: x
    f_mod.one_hot = lambda *args, **kwargs: 0

    torch_mod.nn = nn_mod

    # torchaudio.compliance.kaldi
    ta_mod = types.ModuleType("torchaudio")
    ta_comp = types.ModuleType("torchaudio.compliance")
    ta_kaldi = types.ModuleType("torchaudio.compliance.kaldi")
    ta_kaldi.fbank = lambda *args, **kwargs: 0
    ta_comp.kaldi = ta_kaldi
    ta_mod.compliance = ta_comp

    # librosa.filters.mel
    librosa_mod = types.ModuleType("librosa")
    librosa_filters = types.ModuleType("librosa.filters")
    librosa_filters.mel = lambda **kwargs: 0
    librosa_mod.filters = librosa_filters

    # package chain for relative imports
    pkg_qwen = types.ModuleType("qwen_tts")
    pkg_qwen.__path__ = []
    pkg_core = types.ModuleType("qwen_tts.core")
    pkg_core.__path__ = []
    pkg_tok = types.ModuleType("qwen_tts.core.tokenizer_25hz")
    pkg_tok.__path__ = []
    pkg_vq = types.ModuleType("qwen_tts.core.tokenizer_25hz.vq")
    pkg_vq.__path__ = []

    core_vq_mod = types.ModuleType("qwen_tts.core.tokenizer_25hz.vq.core_vq")
    core_vq_mod.DistributedGroupResidualVectorQuantization = type(
        "DistributedGroupResidualVectorQuantization",
        (),
        {},
    )

    whisper_mod = types.ModuleType("qwen_tts.core.tokenizer_25hz.vq.whisper_encoder")
    whisper_mod.WhisperEncoder = type("WhisperEncoder", (Module,), {})
    whisper_mod.Conv1d = type("Conv1d", (Module,), {})
    whisper_mod.ConvTranspose1d = type("ConvTranspose1d", (Module,), {})

    return {
        "torch": torch_mod,
        "torch.nn": nn_mod,
        "torch.nn.functional": f_mod,
        "torchaudio": ta_mod,
        "torchaudio.compliance": ta_comp,
        "torchaudio.compliance.kaldi": ta_kaldi,
        "librosa": librosa_mod,
        "librosa.filters": librosa_filters,
        "qwen_tts": pkg_qwen,
        "qwen_tts.core": pkg_core,
        "qwen_tts.core.tokenizer_25hz": pkg_tok,
        "qwen_tts.core.tokenizer_25hz.vq": pkg_vq,
        "qwen_tts.core.tokenizer_25hz.vq.core_vq": core_vq_mod,
        "qwen_tts.core.tokenizer_25hz.vq.whisper_encoder": whisper_mod,
    }


@contextmanager
def loaded_speech_vq_module(extra_modules=None):
    extra_modules = extra_modules or {}
    overrides = build_speech_vq_stubs()
    overrides.update(extra_modules)

    with temporary_modules(overrides):
        spec = importlib.util.spec_from_file_location(
            "qwen_tts.core.tokenizer_25hz.vq.speech_vq",
            SPEECH_VQ_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["qwen_tts.core.tokenizer_25hz.vq.speech_vq"] = module
        try:
            assert spec.loader is not None
            spec.loader.exec_module(module)
            yield module
        finally:
            sys.modules.pop("qwen_tts.core.tokenizer_25hz.vq.speech_vq", None)


class _NoOpContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestSpeechVqLazyImports(unittest.TestCase):
    def test_module_import_does_not_require_optional_dependencies(self):
        with blocked_imports({"onnxruntime", "sox"}):
            with loaded_speech_vq_module() as module:
                self.assertTrue(hasattr(module, "XVectorExtractor"))

    def test_xvector_extractor_raises_when_onnxruntime_missing(self):
        with loaded_speech_vq_module() as module:
            with blocked_imports({"onnxruntime"}):
                with self.assertRaises(RuntimeError) as ctx:
                    module.XVectorExtractor("dummy.onnx")
        self.assertIn("onnxruntime is required", str(ctx.exception))

    def test_xvector_extractor_raises_when_sox_missing(self):
        onnxruntime_mod = types.ModuleType("onnxruntime")

        class SessionOptions:
            def __init__(self):
                self.graph_optimization_level = None
                self.intra_op_num_threads = None

        class GraphOptimizationLevel:
            ORT_ENABLE_ALL = 99

        class InferenceSession:
            def __init__(self, *_args, **_kwargs):
                pass

        onnxruntime_mod.SessionOptions = SessionOptions
        onnxruntime_mod.GraphOptimizationLevel = GraphOptimizationLevel
        onnxruntime_mod.InferenceSession = InferenceSession

        with loaded_speech_vq_module({"onnxruntime": onnxruntime_mod}) as module:
            with blocked_imports({"sox"}):
                with self.assertRaises(RuntimeError) as ctx:
                    module.XVectorExtractor("dummy.onnx")
        self.assertIn("python-sox and SoX are required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
