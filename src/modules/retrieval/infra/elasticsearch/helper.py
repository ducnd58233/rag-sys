def _metadata_as_str_map(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    
    return {
        key: str(value) for key, value in raw.items()
    }
