"""Stage 6: the bench request sets (serve/bench_data.py), the quantized checkpoint checks
(serve/quantize.py) and the served smoke's near-tie rule (serve/served_check.py).

The quantized configs are checked again once pulled (results/serve/quantize/<variant>/): the YaRN key,
the untied lm_head and the scheme, so a re-save that dropped any of them fails here as well as on
Modal."""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for d in ("serve", "eval", "train"):
    sys.path.insert(0, str(REPO / d))
PULLED = sorted((REPO / "results/serve/quantize").glob("*/config.json"))


@pytest.fixture(scope="module")
def bd():
    import bench_data

    return bench_data


@pytest.fixture(scope="module")
def sets(bd):
    return bd.build(REPO / "eval/tasks")


def context(prompt: str) -> str:
    return prompt.rsplit("\n\nQuestion: ", 1)[0]


def test_sets_match_the_manifest(bd, sets):
    want = {k: v["sha256"] for k, v in json.loads(bd.MANIFEST.read_text())["files"].items()}
    got = {k: hashlib.sha256(bd.dump(v).encode()).hexdigest() for k, v in sets.items()}
    assert got == want


def test_unique_mix_and_caps(sets):
    rows = sets["unique"]
    assert Counter(r["task"] for r in rows) == {
        "domain_qa": 216,
        "grounded": 108,
        "adversarial": 36,
    }
    assert len({r["prompt"] for r in rows}) == len(rows) == 360
    caps = {"domain_qa": 64, "grounded": 300, "adversarial": 200}  # prompts.GEN + CHAT_GEN
    assert all(r["output_tokens"] == caps[r["task"]] for r in rows)


def test_prompts_are_untemplated(sets):
    """The server applies the chat template; a templated prompt would be templated twice."""
    for rows in sets.values():
        for r in rows:
            assert not re.search(r"\[/?INST\]|<s>|</s>|\[SYSTEM_PROMPT\]", r["prompt"]), r["id"]


def test_rag_shares_contexts_and_stays_answerable(sets):
    unique, rag = sets["grounded_unique"], sets["grounded_rag"]
    assert [r["id"] for r in unique] == [r["id"] for r in rag]  # same questions, same order
    assert len({context(r["prompt"]) for r in unique}) == 108  # the eval shares no passages
    assert sorted(Counter(context(r["prompt"]) for r in rag).values()) == [4] * 27
    tasks = {
        t["id"]: t
        for t in map(json.loads, (REPO / "eval/tasks/grounded.jsonl").read_text().splitlines())
    }
    for r in rag:
        t = tasks[r["id"]]
        assert r["prompt"].endswith(f"\n\nQuestion: {t['question']}\nAnswer:")  # question last
        (gold,) = [c for c in t["context"] if c["chunk_id"] in t["gold_chunk_ids"]]
        assert f"[{gold['chunk_id']}]\n{gold['text']}" in context(r["prompt"])


def test_ignore_list_keeps_the_tower_projector_and_lm_head():
    """llm-compressor's "re:" targets are matched from the start of the module name."""
    from quantize import IGNORE

    pats = [re.compile(p.removeprefix("re:")) for p in IGNORE]
    ignored = [
        "lm_head",
        "model.vision_tower.transformer.layers.0.attention.q_proj",
        "model.multi_modal_projector.linear_1",
        "model.multi_modal_projector.patch_merger.merging_layer",
    ]
    quantized = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.33.mlp.down_proj",
    ]
    assert all(any(p.match(n) for p in pats) for n in ignored)
    assert not any(p.match(n) for p in pats for n in quantized)


def write_ckpt(d: Path, names: list[str], config: dict) -> None:
    import torch
    from safetensors.torch import save_file

    d.mkdir(parents=True)
    save_file({n: torch.zeros(1) for n in names}, str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps(config))


BASE_NAMES = [
    "language_model.lm_head.weight",
    "language_model.model.embed_tokens.weight",
    "language_model.model.layers.0.mlp.up_proj.weight",
    "vision_tower.transformer.layers.0.attention.q_proj.weight",
]
W4 = ["language_model.model.layers.0.mlp.up_proj"]


def quantized_config(**text) -> dict:
    rope = {"rope_type": "yarn", "apply_yarn_scaling": False}
    w = {"num_bits": 4, "group_size": 128, "symmetric": True, "type": "int"}
    return {
        "tie_word_embeddings": False,
        "text_config": {"tie_word_embeddings": False, "rope_parameters": rope, **text},
        "quantization_config": {"config_groups": {"group_0": {"weights": w}}},
    }


def w4_names(extra: tuple[str, ...] = ()) -> list[str]:
    names = [n for n in BASE_NAMES if not n.startswith(W4[0])] + list(extra)
    return names + [f"{W4[0]}.{s}" for s in ("weight_packed", "weight_scale", "weight_shape")]


def test_check_quantized(tmp_path):
    from quantize import check_quantized

    src = tmp_path / "src"
    write_ckpt(src, BASE_NAMES, {})
    ok = tmp_path / "ok"
    write_ckpt(ok, w4_names(), quantized_config())
    assert check_quantized(src, ok, "w4a16")["quantized_modules"] == 1

    yarn = tmp_path / "yarn"
    write_ckpt(yarn, w4_names(), quantized_config(rope_parameters={"rope_type": "yarn"}))
    with pytest.raises(SystemExit, match="apply_yarn_scaling"):
        check_quantized(src, yarn, "w4a16")

    tied = tmp_path / "tied"
    write_ckpt(tied, w4_names(), quantized_config(tie_word_embeddings=True))
    with pytest.raises(SystemExit, match="tie_word_embeddings"):
        check_quantized(src, tied, "w4a16")

    head = tmp_path / "head"
    write_ckpt(head, w4_names(("language_model.lm_head.weight_scale",)), quantized_config())
    with pytest.raises(SystemExit, match="ignored modules were quantized"):
        check_quantized(src, head, "w4a16")


@pytest.mark.skipif(not PULLED, reason="no quantized configs pulled")
def test_pulled_quantized_configs():
    for path in PULLED:
        cfg = json.loads(path.read_text())
        text = cfg["text_config"]
        assert text["rope_parameters"]["apply_yarn_scaling"] is False, path
        assert cfg["tie_word_embeddings"] is False and text["tie_word_embeddings"] is False, path
        meta = json.loads((path.parent / "quantize_meta.json").read_text())
        weights = [g["weights"] for g in cfg["quantization_config"]["config_groups"].values()]
        if meta["scheme"] == "w4a16":
            assert all(
                (w["num_bits"], w["group_size"], w["symmetric"]) == (4, 128, True) for w in weights
            )
        else:
            assert all((w["num_bits"], w["type"]) == (8, "float") for w in weights)
        assert meta["source_digest"].startswith("e686ca7b"), "not quantized from dpo-strict"


def tok(token: str, lp: float, *alts: tuple[str, float]) -> dict:
    return {"token": token, "logprob": lp, "top_logprobs": [{"token": token, "logprob": lp}]
            + [{"token": t, "logprob": v} for t, v in alts]}  # fmt: skip


def test_first_divergence():
    from served_check import first_divergence

    served = [tok("The", -0.1, ("A", -2.4)), tok(" beam", -0.70, (" girder", -0.74))]
    near = first_divergence(served, "The girder carries")
    assert near["index"] == 1 and near["margin"] == pytest.approx(0.04)
    assert first_divergence(served, "The column")["margin"] is None  # not in the server's top 2
    assert first_divergence(served, "A beam")["margin"] == pytest.approx(2.3)
    assert first_divergence(served, "The beam is")["index"] == 2  # served text is a prefix
