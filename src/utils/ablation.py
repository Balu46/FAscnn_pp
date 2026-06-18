from dataclasses import dataclass
from typing import Dict, List


_TOKEN_ALIASES: Dict[str, str] = {
    "no_attn": "no_attn",
    "no_attention": "no_attn",
    "noattn": "no_attn",
    "no_fa1": "no_fa1",
    "no_fa2": "no_fa2",
    "no_fa3": "no_fa3",
    "no_branch": "no_branch",
    "no_second_branch": "no_branch",
    "no_extra_branch": "no_branch",
    "none": "none",
    "baseline": "none",
}


@dataclass(frozen=True)
class AblationConfig:
    use_fa1: bool = True
    use_fa2: bool = True
    use_fa3: bool = True
    use_second_branch: bool = True

    def id(self) -> str:
        tokens: List[str] = []
        if not self.use_second_branch:
            tokens.append("nobranch")

        if not (self.use_fa1 and self.use_fa2 and self.use_fa3):
            if not self.use_fa1 and not self.use_fa2 and not self.use_fa3:
                tokens.append("noattn")
            else:
                if not self.use_fa1:
                    tokens.append("no_fa1")
                if not self.use_fa2:
                    tokens.append("no_fa2")
                if not self.use_fa3:
                    tokens.append("no_fa3")

        return "__".join(tokens)


def parse_ablation_spec(spec: str) -> AblationConfig:
    if not spec:
        return AblationConfig()

    raw_tokens = [token.strip().lower() for token in spec.split(",") if token.strip()]
    tokens: List[str] = []
    for raw in raw_tokens:
        if raw not in _TOKEN_ALIASES:
            raise ValueError(f"Unknown ablation token: {raw}")
        normalized = _TOKEN_ALIASES[raw]
        if normalized != "none":
            tokens.append(normalized)

    disable_all_attn = "no_attn" in tokens
    use_fa1 = not (disable_all_attn or "no_fa1" in tokens)
    use_fa2 = not (disable_all_attn or "no_fa2" in tokens)
    use_fa3 = not (disable_all_attn or "no_fa3" in tokens)

    use_second_branch = "no_branch" not in tokens
    if not use_second_branch:
        use_fa2 = False
        use_fa3 = False

    return AblationConfig(
        use_fa1=use_fa1,
        use_fa2=use_fa2,
        use_fa3=use_fa3,
        use_second_branch=use_second_branch,
    )


def make_run_tag(model_name: str, ablation: AblationConfig) -> str:
    ablation_id = ablation.id()
    if ablation_id:
        return f"{model_name}__{ablation_id}"
    return model_name
