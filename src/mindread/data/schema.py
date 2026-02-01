"""DSTC2 schema definitions and slot ontology."""

from dataclasses import dataclass, field
from typing import Literal

# DSTC2 Informable Slots (user can inform these)
INFORMABLE_SLOTS = ["food", "area", "pricerange"]

# DSTC2 Requestable Slots (user can request these)
REQUESTABLE_SLOTS = ["food", "area", "pricerange", "phone", "postcode", "addr", "name"]

# Slot value ontology for DSTC2
SLOT_VALUES: dict[str, list[str]] = {
    "food": [
        "afghan", "african", "asian oriental", "australasian", "australian",
        "austrian", "barbeque", "basque", "belgian", "bistro", "brazilian",
        "british", "canapes", "cantonese", "caribbean", "catalan", "chinese",
        "christmas", "corsica", "creative", "crossover", "cuban", "danish",
        "english", "eritrean", "european", "french", "fusion", "gastropub",
        "german", "greek", "halal", "hungarian", "indian", "indonesian",
        "international", "irish", "italian", "jamaican", "japanese", "korean",
        "kosher", "latin american", "lebanese", "light bites", "malaysian",
        "mediterranean", "mexican", "middle eastern", "modern american",
        "modern eclectic", "modern european", "modern global", "moroccan",
        "new zealand", "north african", "north american", "north indian",
        "northern european", "panasian", "persian", "polish", "polynesian",
        "portuguese", "romanian", "russian", "scandinavian", "scottish",
        "seafood", "singaporean", "south african", "south indian",
        "spanish", "sri lankan", "steakhouse", "swedish", "swiss", "thai",
        "the americas", "traditional", "turkish", "tuscan", "unusual",
        "vegetarian", "venetian", "vietnamese", "welsh", "world",
    ],
    "area": [
        "centre", "north", "south", "east", "west",
    ],
    "pricerange": [
        "cheap", "moderate", "expensive",
    ],
}

# Special values that can appear in any slot
SPECIAL_VALUES = ["dontcare"]


@dataclass
class DialogueAct:
    """Represents a single dialogue act."""

    act_type: str
    slots: list[tuple[str, str]] = field(default_factory=list)

    def __repr__(self) -> str:
        if self.slots:
            slot_str = ", ".join(f"{k}={v}" for k, v in self.slots)
            return f"{self.act_type}({slot_str})"
        return self.act_type


@dataclass
class Turn:
    """Represents a single turn in a dialogue."""

    turn_index: int
    user_utterance: str
    system_utterance: str
    user_acts: list[DialogueAct]
    system_acts: list[DialogueAct]
    belief_state: dict[str, str]  # Cumulative belief state at this turn


@dataclass
class Dialogue:
    """Represents a complete dialogue session."""

    dialogue_id: str
    turns: list[Turn]
    goal: dict[str, str]  # User's goal for this dialogue

    @property
    def num_turns(self) -> int:
        return len(self.turns)


@dataclass
class DialogueStateLabel:
    """Label for dialogue state tracking at a single turn."""

    food: str | None = None
    area: str | None = None
    pricerange: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "food": self.food,
            "area": self.area,
            "pricerange": self.pricerange,
        }

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "DialogueStateLabel":
        return cls(
            food=d.get("food"),
            area=d.get("area"),
            pricerange=d.get("pricerange"),
        )

    def matches(self, other: "DialogueStateLabel") -> bool:
        """Check if this state matches another (joint goal accuracy)."""
        return (
            self.food == other.food
            and self.area == other.area
            and self.pricerange == other.pricerange
        )


Split = Literal["train", "dev", "test"]
