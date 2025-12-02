"""Text cleaning module for post-processing transcriptions."""

import time

from llama_cpp import Llama

from .config import Config


class TextCleaner:
    """Text cleaner using language model for cleaning raw transcriptions."""

    def __init__(self, config: Config):
        """
        Initialize text cleaner with language model.

        Args:
            config: Configuration object with model paths and settings
        """
        self.config = config
        self._llama: Llama | None = None
        self._model_loaded = False

        # Validate text cleaner model exists
        if not self.config.text_cleaner_model_path.exists():
            raise FileNotFoundError(
                f"Text cleaner model not found: {self.config.text_cleaner_model_path}"
            )

    def load_model(self) -> bool:
        """
        Load the text cleaning model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._model_loaded:
            return True

        try:
            print("🧹 Loading text cleaning model...")
            self._llama = Llama(
                model_path=str(self.config.text_cleaner_model_path), verbose=False
            )

            self._model_loaded = True
            print("✅ Text cleaning model loaded successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to load text cleaning model: {e}")
            return False

    def clean_text(self, raw_text: str) -> str:
        """
        Clean raw transcription text using language model.

        Args:
            raw_text: Raw transcription text to clean

        Returns:
            Cleaned transcription text

        Raises:
            RuntimeError: If model is not loaded or cleaning fails
        """
        if not raw_text or not raw_text.strip():
            return raw_text

        if not self._model_loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load text cleaning model")

        try:
            # Get messages for chat completion
            messages = self._get_messages(raw_text)

            # Generate cleaned text
            start_time = time.time()

            # Reset model state if available
            if hasattr(self._llama, "reset"):
                self._llama.reset()

            response = self._llama.create_chat_completion(
                messages=messages,
                # TODO: extract these parameters to config
                temperature=0.1,
                min_p=0.15,
                repeat_penalty=1.05,
                max_tokens=512,
            )

            # response = self._llama.create_chat_completion(messages=messages, temperature=0.1, min_p=0.15, repeat_penalty=1.05)

            # Extract cleaned text from response
            cleaned_text = self._extract_cleaned_text(response)

            elapsed_time = time.time() - start_time
            # print(f"🧹 Text cleaning completed in {elapsed_time:.1f}s")

            return cleaned_text

        except Exception as e:
            print(f"⚠️ Text cleaning failed: {e}")
            print("📝 Falling back to raw transcription")
            breakpoint()
            return raw_text

    def _get_messages(self, raw_text: str) -> list[dict[str, str]]:
        """
        Get messages for chat completion API.

        Args:
            raw_text: Raw text to be cleaned

        Returns:
            List of message dictionaries for chat completion
        """
        return [
            {"role": "system", "content": self.config.text_cleaner_system_prompt},
            {
                "role": "user",
                "content": self.config.text_cleaner_user_prompt.format(
                    raw_text=raw_text
                ),
            },
        ]

    def _extract_cleaned_text(self, response: dict) -> str:
        """
        Extract cleaned text from chat completion response.

        Args:
            response: Response from create_chat_completion

        Returns:
            Cleaned text string
        """
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                text = choice["message"]["content"].strip()

                # Clean up extra whitespace
                text = " ".join(text.split())

                return text

        return ""

    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self._llama:
            # llama-cpp-python handles cleanup automatically
            pass
        self._model_loaded = False


def create_text_cleaner(config: Config) -> TextCleaner | None:
    """
    Create text cleaner with fallback for missing dependencies or model files.

    Args:
        config: Configuration object

    Returns:
        TextCleaner instance or None if creation fails
    """
    try:
        return TextCleaner(config)
    except Exception as e:
        print(f"⚠️ Text cleaning not available: {e}")
        return None
