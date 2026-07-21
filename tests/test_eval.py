import json

from s2t_bench.eval.concurrency import run_concurrent
from s2t_bench.eval.hardware import HardwareMonitor
from s2t_bench.eval.gemma import schema, scorer
from s2t_bench.eval.gemma.samples import load_samples


# ---- schema + samples ----

def test_all_samples_conform_to_schema():
    for s in load_samples():
        assert schema.is_valid(s.expected), s.id


def test_schema_rejects_bad_types():
    bad = {f: "" for f in schema.STRING_FIELDS}
    bad.update({f: "notbool" for f in schema.BOOL_FIELDS})
    bad.update({f: "notlist" for f in schema.COMPONENT_LIST_FIELDS})
    errs = schema.validate(bad)
    assert any("boolean" in e for e in errs)
    assert any("not a list" in e for e in errs)


# ---- scorer ----

def test_extract_json_handles_fences_and_prose():
    assert scorer.extract_json('```json\n{"a":1}\n```') == {"a": 1}
    assert scorer.extract_json('here you go: {"a": 2} thanks') == {"a": 2}
    assert scorer.extract_json("not json") is None


def test_score_json_perfect_and_wrong():
    sample = load_samples()[0]
    perfect = scorer.score_json(sample.expected, sample.expected)
    assert perfect["schema_valid"] and perfect["field_accuracy"] == 1.0

    wrong = dict(sample.expected)
    wrong["reported_issue_correct_flag"] = not wrong["reported_issue_correct_flag"]
    wrong["rc_hl_category"] = "TotallyWrong"
    wrong["faulty_components"] = []
    s = scorer.score_json(sample.expected, wrong)
    assert s["field_accuracy"] < 1.0


def test_component_matching_by_sku():
    exp = [{"item": "ONT", "sku": "ONT-5678"}]
    assert scorer._components_match(exp, [{"item": "x", "sku": "ont-5678"}]) == 1.0
    assert scorer._components_match(exp, []) == 0.0
    assert scorer._components_match([], []) == 1.0


# ---- concurrency ----

def test_run_concurrent_records_latency_and_errors():
    def fn(x):
        if x == 3:
            raise ValueError("boom")
        return x * 2

    res = run_concurrent(fn, [1, 2, 3, 4], concurrency=2)
    assert res.num_tasks == 4
    assert res.num_errors == 1
    assert len([o for o in res.outcomes if o.ok]) == 3
    assert res.throughput_per_s > 0


# ---- hardware ----

def test_hardware_monitor_collects_samples():
    with HardwareMonitor(interval_s=0.05) as hw:
        sum(i * i for i in range(200000))
    s = hw.summary()
    assert s.num_samples >= 1
    assert s.num_cpus >= 1
    assert s.total_ram_mb > 0


# ---- gemma driver with injected client ----

def test_gemma_eval_driver_with_fake_client():
    from s2t_bench.eval.gemma import gemma_eval

    samples = load_samples()[:4]

    def fake(model, prompt, system=None, fmt=None, host=None, **kw):
        for s in samples:
            if s.corrected_text[:25] in prompt:
                return json.dumps(s.expected)
            if s.raw_text[:25] in prompt:
                return s.corrected_text
        return "{}"

    res = gemma_eval.evaluate_model(
        "fake", tasks=["correction", "json"], concurrency=2,
        samples=samples, generate_fn=fake,
    )
    assert res["json"]["quality"]["schema_valid_rate"] == 1.0
    assert res["correction"]["quality"]["wer"] == 0.0
    assert "hardware" in res and res["json"]["perf"]["num_tasks"] == 4


# ---- whisper driver with injected engine ----

def test_whisper_eval_driver_with_stub_engine():
    from dataclasses import dataclass
    from s2t_bench.eval import whisper_eval
    from s2t_bench.engines.base import TranscriptionResult

    @dataclass
    class Smp:
        id: str
        audio_path: str
        reference: str

    samples = [Smp("a", "x.wav", "the olt port was down")]

    class Eng:
        def __init__(self, m, c):
            self.m = m
        def _get_model(self):
            pass
        def transcribe(self, path):
            return TranscriptionResult("the olt port was down", self.m, 3.0, 0.6)

    r = whisper_eval.evaluate_model("small", samples, concurrency=1, engine_factory=lambda m, c: Eng(m, c))
    assert r["accuracy"]["avg_wer"] == 0.0
    assert r["speed"]["avg_rtf"] == 0.2
    assert r["hardware"]["num_cpus"] >= 1
