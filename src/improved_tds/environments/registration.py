from __future__ import annotations

from gymnasium.envs.registration import register, registry


def _register(env_id: str, entry_point: str, **kwargs: object) -> None:
    if env_id not in registry:
        register(id=env_id, entry_point=entry_point, **kwargs)


def register_environments() -> None:
    for variant in range(1, 6):
        _register(
            f"ExpScissor{variant}-v0",
            "improved_tds.environments.scissors:ExpScissorEnv",
            kwargs={"variant": variant},
            max_episode_steps=100,
        )
        _register(
            f"ImprovedScissor{variant}-v0",
            "improved_tds.environments.scissors:ExpScissorEnv",
            kwargs={"variant": variant},
            max_episode_steps=100,
        )
    _register(
        "TDS-Trigger-v0",
        "improved_tds.environments.trigger:TriggerEnv",
        max_episode_steps=200,
    )
    _register(
        "TDS-Button-v0",
        "improved_tds.environments.button:ButtonEnv",
        max_episode_steps=200,
    )


register_environments()

