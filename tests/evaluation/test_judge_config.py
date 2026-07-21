from __future__ import annotations

from scripts.evaluation.judge.config import resolve_judge_config

from src.shared.configs.settings import ChatSettings


def test_resolve_judge_config_flags_self_preference_when_no_override_is_set(
    monkeypatch,
) -> None:
    monkeypatch.delenv("EVAL_JUDGE_PROVIDER", raising=False)
    monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)
    app_settings = ChatSettings(provider="ollama", model="qwen2.5:1.5b")

    config = resolve_judge_config(app_settings)

    assert config.provider == "ollama"
    assert config.model == "qwen2.5:1.5b"
    assert config.self_preference_risk is True


def test_resolve_judge_config_clears_self_preference_risk_when_model_overridden(
    monkeypatch,
) -> None:
    monkeypatch.delenv("EVAL_JUDGE_PROVIDER", raising=False)
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "a-bigger-judge-model")
    app_settings = ChatSettings(provider="ollama", model="qwen2.5:1.5b")

    config = resolve_judge_config(app_settings)

    assert config.model == "a-bigger-judge-model"
    assert config.self_preference_risk is False


def test_resolve_judge_config_uses_zero_temperature_regardless_of_app_settings() -> (
    None
):
    app_settings = ChatSettings(temperature=0.7)

    config = resolve_judge_config(app_settings)

    assert config.temperature == 0.0
