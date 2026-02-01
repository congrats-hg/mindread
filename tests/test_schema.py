"""Tests for DSTC2 schema definitions."""

import pytest

from mindread.data.schema import (
    INFORMABLE_SLOTS,
    SLOT_VALUES,
    Dialogue,
    DialogueAct,
    DialogueStateLabel,
    Turn,
)


class TestDialogueAct:
    """Tests for DialogueAct dataclass."""

    def test_create_without_slots(self) -> None:
        act = DialogueAct(act_type="hello")
        assert act.act_type == "hello"
        assert act.slots == []

    def test_create_with_slots(self) -> None:
        act = DialogueAct(act_type="inform", slots=[("food", "italian"), ("area", "centre")])
        assert act.act_type == "inform"
        assert len(act.slots) == 2

    def test_repr_without_slots(self) -> None:
        act = DialogueAct(act_type="hello")
        assert repr(act) == "hello"

    def test_repr_with_slots(self) -> None:
        act = DialogueAct(act_type="inform", slots=[("food", "italian")])
        assert "inform" in repr(act)
        assert "food=italian" in repr(act)


class TestDialogueStateLabel:
    """Tests for DialogueStateLabel dataclass."""

    def test_create_empty(self) -> None:
        label = DialogueStateLabel()
        assert label.food is None
        assert label.area is None
        assert label.pricerange is None

    def test_create_with_values(self) -> None:
        label = DialogueStateLabel(food="italian", area="centre", pricerange="cheap")
        assert label.food == "italian"
        assert label.area == "centre"
        assert label.pricerange == "cheap"

    def test_to_dict(self) -> None:
        label = DialogueStateLabel(food="italian", area="centre")
        d = label.to_dict()
        assert d["food"] == "italian"
        assert d["area"] == "centre"
        assert d["pricerange"] is None

    def test_from_dict(self) -> None:
        d = {"food": "italian", "area": "centre", "pricerange": "cheap"}
        label = DialogueStateLabel.from_dict(d)
        assert label.food == "italian"
        assert label.area == "centre"
        assert label.pricerange == "cheap"

    def test_matches_identical(self) -> None:
        label1 = DialogueStateLabel(food="italian", area="centre")
        label2 = DialogueStateLabel(food="italian", area="centre")
        assert label1.matches(label2)

    def test_matches_different(self) -> None:
        label1 = DialogueStateLabel(food="italian", area="centre")
        label2 = DialogueStateLabel(food="chinese", area="centre")
        assert not label1.matches(label2)


class TestTurn:
    """Tests for Turn dataclass."""

    def test_create_turn(self) -> None:
        turn = Turn(
            turn_index=0,
            user_utterance="I want italian food",
            system_utterance="What area?",
            user_acts=[DialogueAct(act_type="inform", slots=[("food", "italian")])],
            system_acts=[DialogueAct(act_type="request", slots=[("area", None)])],
            belief_state={"food": "italian"},
        )
        assert turn.turn_index == 0
        assert turn.user_utterance == "I want italian food"
        assert len(turn.user_acts) == 1
        assert turn.belief_state["food"] == "italian"


class TestDialogue:
    """Tests for Dialogue dataclass."""

    def test_create_dialogue(self) -> None:
        turns = [
            Turn(
                turn_index=0,
                user_utterance="I want italian food",
                system_utterance="What area?",
                user_acts=[],
                system_acts=[],
                belief_state={"food": "italian"},
            ),
            Turn(
                turn_index=1,
                user_utterance="Centre please",
                system_utterance="Here are some options",
                user_acts=[],
                system_acts=[],
                belief_state={"food": "italian", "area": "centre"},
            ),
        ]
        dialogue = Dialogue(
            dialogue_id="test_001",
            turns=turns,
            goal={"food": "italian", "area": "centre"},
        )
        assert dialogue.dialogue_id == "test_001"
        assert dialogue.num_turns == 2
        assert dialogue.goal["food"] == "italian"


class TestSlotOntology:
    """Tests for slot ontology definitions."""

    def test_informable_slots(self) -> None:
        assert "food" in INFORMABLE_SLOTS
        assert "area" in INFORMABLE_SLOTS
        assert "pricerange" in INFORMABLE_SLOTS
        assert len(INFORMABLE_SLOTS) == 3

    def test_slot_values_exist(self) -> None:
        for slot in INFORMABLE_SLOTS:
            assert slot in SLOT_VALUES
            assert len(SLOT_VALUES[slot]) > 0

    def test_area_values(self) -> None:
        areas = SLOT_VALUES["area"]
        assert "centre" in areas
        assert "north" in areas
        assert "south" in areas

    def test_pricerange_values(self) -> None:
        prices = SLOT_VALUES["pricerange"]
        assert "cheap" in prices
        assert "moderate" in prices
        assert "expensive" in prices
