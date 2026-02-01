"""Interactive Gradio demo for Dialogue State Tracking."""

import argparse
import logging
from pathlib import Path
from typing import Any

import torch

from .data.dataset import SLOT_VOCAB
from .data.schema import INFORMABLE_SLOTS

logger = logging.getLogger(__name__)


class DSTDemo:
    """Interactive demo for dialogue state tracking."""

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        tokenizer: Any = None,
        device: torch.device | None = None,
    ):
        """
        Initialize demo.

        Args:
            model: Trained DST model. If None, runs in mock mode.
            tokenizer: HuggingFace tokenizer.
            device: Device for inference.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")

        if model is not None:
            self.model.to(self.device)
            self.model.eval()

        # Dialogue history
        self.reset_dialogue()

    def reset_dialogue(self) -> dict[str, str | None]:
        """Reset dialogue state."""
        self.history: list[tuple[str, str]] = []  # (role, utterance)
        self.current_state = {slot: None for slot in INFORMABLE_SLOTS}
        return self.current_state

    def _build_context(self) -> str:
        """Build context string from dialogue history."""
        context_parts = []
        for role, utterance in self.history[-10:]:  # Keep last 10 turns
            prefix = "[SYS]" if role == "system" else "[USR]"
            context_parts.append(f"{prefix} {utterance}")
        return " ".join(context_parts)

    def _predict(self, user_input: str, system_response: str = "") -> dict[str, str]:
        """Get model predictions for current turn."""
        if self.model is None or self.tokenizer is None:
            # Mock mode - return current state
            return {slot: self.current_state[slot] or "none" for slot in INFORMABLE_SLOTS}

        # Add system response to history if provided
        if system_response:
            self.history.append(("system", system_response))

        # Build input text
        context = self._build_context()
        current = f"[SYS] {system_response} [USR] {user_input}".strip()

        if context:
            text = f"{context} [SEP] {current}"
        else:
            text = current

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        # Get predictions
        with torch.no_grad():
            predictions = self.model.predict(input_ids, attention_mask, token_type_ids)

        # Convert indices to values
        result = {}
        for slot in INFORMABLE_SLOTS:
            idx = predictions[slot][0].item()
            value = SLOT_VOCAB[slot][idx]
            result[slot] = value

        # Add user input to history
        self.history.append(("user", user_input))

        return result

    def process_turn(
        self,
        user_input: str,
        system_response: str = "",
    ) -> tuple[dict[str, str], str]:
        """
        Process a dialogue turn.

        Args:
            user_input: User's utterance.
            system_response: System's response (optional).

        Returns:
            Tuple of (predicted state, formatted state string).
        """
        predictions = self._predict(user_input, system_response)

        # Update current state (carry over values)
        for slot, value in predictions.items():
            if value != "none":
                self.current_state[slot] = value

        # Format state for display
        state_str = self._format_state()

        return dict(self.current_state), state_str

    def _format_state(self) -> str:
        """Format current state for display."""
        lines = ["**Current Belief State:**", ""]
        for slot in INFORMABLE_SLOTS:
            value = self.current_state[slot]
            status = value if value else "not specified"
            lines.append(f"- **{slot.capitalize()}**: {status}")
        return "\n".join(lines)


def create_gradio_interface(demo: DSTDemo) -> Any:
    """Create Gradio interface."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio not installed. Run: pip install gradio")

    def process_input(
        user_input: str,
        system_response: str,
        history: list,
    ) -> tuple[list, str, str, str, str]:
        """Process user input and update state."""
        if not user_input.strip():
            return history, "", "", "", ""

        state, state_str = demo.process_turn(user_input, system_response)

        # Update chat history
        if system_response:
            history.append(("System: " + system_response, None))
        history.append((None, "User: " + user_input))

        return (
            history,
            state_str,
            state.get("food") or "not specified",
            state.get("area") or "not specified",
            state.get("pricerange") or "not specified",
        )

    def reset() -> tuple[list, str, str, str, str, str, str]:
        """Reset dialogue."""
        demo.reset_dialogue()
        return [], "**Dialogue reset. Start a new conversation.**", "", "", "", "", ""

    with gr.Blocks(title="MindRead - Dialogue State Tracking") as interface:
        gr.Markdown(
            """
            # MindRead: Dialogue State Tracking Demo

            This demo shows a BERT-based dialogue state tracker for restaurant information.
            Enter user utterances to see how the model tracks the user's goals.

            **Tracked Slots:**
            - **Food**: Type of cuisine (e.g., italian, chinese, indian)
            - **Area**: Location (centre, north, south, east, west)
            - **Price Range**: Budget (cheap, moderate, expensive)
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Dialogue History", height=400)

                with gr.Row():
                    system_input = gr.Textbox(
                        label="System Response (optional)",
                        placeholder="Enter system's response...",
                    )

                with gr.Row():
                    user_input = gr.Textbox(
                        label="User Input",
                        placeholder="Enter user's utterance (e.g., 'I want italian food')",
                    )

                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    reset_btn = gr.Button("Reset Dialogue")

            with gr.Column(scale=1):
                state_display = gr.Markdown(
                    value="**Start a dialogue to see the belief state.**",
                    label="Belief State",
                )

                gr.Markdown("### Individual Slots")
                food_output = gr.Textbox(label="Food", interactive=False)
                area_output = gr.Textbox(label="Area", interactive=False)
                price_output = gr.Textbox(label="Price Range", interactive=False)

        # Example inputs
        gr.Markdown("### Example Inputs")
        gr.Examples(
            examples=[
                ["I'm looking for a cheap restaurant", "Hello, how can I help you?"],
                ["Italian food please", "What kind of food would you like?"],
                ["In the centre of town", "Which area would you prefer?"],
                ["I don't care about the price", "What's your budget?"],
                ["Actually, make it Chinese instead", "I found several Italian restaurants."],
            ],
            inputs=[user_input, system_input],
        )

        # Event handlers
        submit_btn.click(
            fn=process_input,
            inputs=[user_input, system_input, chatbot],
            outputs=[chatbot, state_display, food_output, area_output, price_output],
        ).then(
            fn=lambda: ("", ""),
            outputs=[user_input, system_input],
        )

        user_input.submit(
            fn=process_input,
            inputs=[user_input, system_input, chatbot],
            outputs=[chatbot, state_display, food_output, area_output, price_output],
        ).then(
            fn=lambda: ("", ""),
            outputs=[user_input, system_input],
        )

        reset_btn.click(
            fn=reset,
            outputs=[
                chatbot,
                state_display,
                food_output,
                area_output,
                price_output,
                user_input,
                system_input,
            ],
        )

    return interface


def main() -> None:
    """CLI entry point for demo."""
    parser = argparse.ArgumentParser(description="Launch DST demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (runs in mock mode if not provided)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Pre-trained model name",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the demo on",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load model if checkpoint provided
    model = None
    tokenizer = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        from transformers import AutoTokenizer

        from .evaluation.evaluate import load_model

        model, tokenizer = load_model(
            Path(args.checkpoint),
            model_name=args.model_name,
            device=device,
        )
        logger.info(f"Loaded model from {args.checkpoint}")
    else:
        logger.warning("No checkpoint provided. Running in mock mode.")

    # Create demo
    demo = DSTDemo(model=model, tokenizer=tokenizer, device=device)
    interface = create_gradio_interface(demo)

    # Launch
    interface.launch(
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
