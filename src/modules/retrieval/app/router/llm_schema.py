from pydantic import BaseModel, Field


class RouterStrategySchema(BaseModel):
    name: str = Field(min_length=1)
    weight: float = Field(gt=0.0)
    query: str | None = Field(default=None)
    as_of: str | None = Field(default=None)


class RouterPlanSchema(BaseModel):
    strategies: list[RouterStrategySchema] = Field(default_factory=list)
    reason: str = Field(default="")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
