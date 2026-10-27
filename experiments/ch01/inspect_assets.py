"""E01：不加载模型，直接检查模型目录里的“资产”：文件、config、safetensors 头、tokenizer。

运行（在仓库根目录）：
    .venv/bin/python -m experiments.ch01.inspect_assets
"""
import hashlib
import json
from collections import Counter

import transformers
from safetensors import safe_open
from transformers import AutoTokenizer

from experiments.common.config import MODEL_DIR
from experiments.common.evidence import cli_command, write_run
from experiments.common.fetch_model import ensure_model


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ensure_model()
    files = {p.name: p.stat().st_size for p in sorted(MODEL_DIR.iterdir()) if p.is_file()}
    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    gen_cfg = json.loads((MODEL_DIR / "generation_config.json").read_text())

    shapes, dtypes = {}, Counter()
    with safe_open(MODEL_DIR / "model.safetensors", "pt") as f:
        for name in f.keys():
            sl = f.get_slice(name)
            shapes[name] = list(sl.get_shape())
            dtypes[sl.get_dtype()] += 1
    n_params = sum(eval("*".join(map(str, s))) if s else 1 for s in shapes.values())

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    chat = tok.apply_chat_template([{"role": "user", "content": "Hi"}], tokenize=False,
                                   add_generation_prompt=True)
    sample_ids = tok("Hello world", add_special_tokens=False)["input_ids"]

    embed = shapes["model.embed_tokens.weight"]
    layer_ids = {int(n.split(".")[2]) for n in shapes if n.startswith("model.layers.")}
    print(f"files: {files}")
    print(f"architectures={cfg['architectures']} hidden={cfg['hidden_size']} layers={cfg['num_hidden_layers']} "
          f"heads={cfg['num_attention_heads']} kv_heads={cfg['num_key_value_heads']} vocab={cfg['vocab_size']}")
    print(f"tensors={len(shapes)} dtypes={dict(dtypes)} params={n_params:,} tie_word_embeddings={cfg['tie_word_embeddings']}")
    print(f"embed_tokens={embed} has_lm_head={'lm_head.weight' in shapes}")
    print(f"tokenizer vocab={len(tok)} eos={tok.eos_token!r} 'Hello world'->{sample_ids}")
    print(f"chat template output:\n{chat}")

    assertions = {
        "weights_file_is_safetensors": "model.safetensors" in files,
        "embedding_rows_equal_vocab_size": embed[0] == cfg["vocab_size"],
        "layer_count_matches_config": len(layer_ids) == cfg["num_hidden_layers"],
        "tied_embeddings_have_no_lm_head": (not cfg["tie_word_embeddings"]) or "lm_head.weight" not in shapes,
        "tokenizer_roundtrip": tok.decode(sample_ids) == "Hello world",
        "chat_template_present": "<|im_start|>" in chat,
    }
    status = "PASS" if all(assertions.values()) else "FAIL"
    path = write_run(
        "ch01", "assets", status, assertions,
        {"transformers": transformers.__version__}, "none (no model loaded)", cli_command(),
        extra={"files_bytes": files, "weights_sha256": sha256(MODEL_DIR / "model.safetensors"),
               "num_tensors": len(shapes), "num_parameters": n_params, "tensor_dtypes": dict(dtypes),
               "config": {k: cfg[k] for k in ("architectures", "hidden_size", "num_hidden_layers",
                                              "num_attention_heads", "num_key_value_heads",
                                              "vocab_size", "tie_word_embeddings", "torch_dtype")
                          if k in cfg},
               "generation_config": gen_cfg, "chat_template_example": chat},
    )
    print(f"{status} -> {path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
