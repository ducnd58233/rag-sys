from __future__ import annotations


def extract_element_metadata(element: object) -> dict[str, str]:
    metadata: dict[str, str] = {}
    category = getattr(element, "category", None)
    if category is not None:
        metadata["element_category"] = str(category)

    raw_metadata = getattr(element, "metadata", None)
    if raw_metadata is not None and hasattr(raw_metadata, "to_dict"):
        for key, value in raw_metadata.to_dict().items():
            if value is not None:
                metadata[str(key)] = str(value)
    return metadata