from unstructured.documents.elements import Element
from src.modules.ingestion.domain.models import ExtractedElement

def to_domain(element: Element) -> ExtractedElement:
    category = getattr(element, "category", "Unknown")
    metadata: dict[str, str] = {}
    raw_metadata = getattr(element, "metadata", None)
    if raw_metadata is not None and hasattr(raw_metadata, "to_dict"):
        for key, value in raw_metadata.to_dict().items():
            if value is not None:
                metadata[str(key)] = str(value)
    return ExtractedElement(
        text=str(element).strip(),
        category=str(category),
        metadata=metadata,
    )

def to_unstructured(element: ExtractedElement) -> Element:
    return Element(
        text=element.text,
        category=element.category,
        metadata=element.metadata,
    )
