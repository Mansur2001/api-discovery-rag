"""Unified LLM interface supporting 4 providers."""

from abc import ABC, abstractmethod

from config import LLM_MODELS, get_api_key


class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class LLMAuthError(LLMError):
    """Invalid or missing API key."""
    pass


class LLMConnectionError(LLMError):
    """Network or API connectivity errors."""
    pass


class LLMRateLimitError(LLMError):
    """Rate limit exceeded."""
    pass


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Send prompt, return generated text."""
        ...

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """Check if this provider is configured and reachable.

        Returns (True, "") if ready, (False, "reason") if not.
        """
        ...


class OpenAIProvider(LLMProvider):
    """GPT-4o via openai Python SDK."""

    def __init__(self, api_key: str, model_id: str = "gpt-4o"):
        self._api_key = api_key
        self._model_id = model_id

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        if not self._api_key:
            raise LLMAuthError("OPENAI_API_KEY is not set.")
        try:
            from openai import OpenAI, APIConnectionError, RateLimitError, AuthenticationError

            client = OpenAI(api_key=self._api_key)
            response = client.chat.completions.create(
                model=self._model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content or ""
        except AuthenticationError:
            raise LLMAuthError("Invalid OpenAI API key.")
        except RateLimitError:
            raise LLMRateLimitError("OpenAI rate limit exceeded. Try again later.")
        except APIConnectionError:
            raise LLMConnectionError("Could not connect to OpenAI API.")
        except ImportError:
            raise LLMError("openai package not installed. Run: pip install openai")
        except Exception as e:
            raise LLMError(f"OpenAI error: {e}")

    def is_available(self) -> tuple[bool, str]:
        if not self._api_key:
            return False, "OPENAI_API_KEY not set"
        return True, ""


class AnthropicProvider(LLMProvider):
    """Claude 3.5 Sonnet via anthropic Python SDK."""

    def __init__(self, api_key: str, model_id: str = "claude-3-5-sonnet-20241022"):
        self._api_key = api_key
        self._model_id = model_id

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        if not self._api_key:
            raise LLMAuthError("ANTHROPIC_API_KEY is not set.")
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self._api_key)
            response = client.messages.create(
                model=self._model_id,
                max_tokens=max_tokens,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text if response.content else ""
        except anthropic.AuthenticationError:
            raise LLMAuthError("Invalid Anthropic API key.")
        except anthropic.RateLimitError:
            raise LLMRateLimitError("Anthropic rate limit exceeded. Try again later.")
        except anthropic.APIConnectionError:
            raise LLMConnectionError("Could not connect to Anthropic API.")
        except ImportError:
            raise LLMError("anthropic package not installed. Run: pip install anthropic")
        except Exception as e:
            raise LLMError(f"Anthropic error: {e}")

    def is_available(self) -> tuple[bool, str]:
        if not self._api_key:
            return False, "ANTHROPIC_API_KEY not set"
        return True, ""


class GoogleProvider(LLMProvider):
    """Gemini 1.5 Pro via google-generativeai SDK."""

    def __init__(self, api_key: str, model_id: str = "gemini-1.5-pro"):
        self._api_key = api_key
        self._model_id = model_id

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        if not self._api_key:
            raise LLMAuthError("GOOGLE_API_KEY is not set.")
        try:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            model = genai.GenerativeModel(self._model_id)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.3,
                ),
            )
            return response.text or ""
        except ImportError:
            raise LLMError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )
        except Exception as e:
            err_str = str(e).lower()
            if "api key" in err_str or "authentication" in err_str:
                raise LLMAuthError("Invalid Google API key.")
            if "quota" in err_str or "rate" in err_str:
                raise LLMRateLimitError("Google API rate limit exceeded.")
            if "connect" in err_str or "network" in err_str:
                raise LLMConnectionError("Could not connect to Google API.")
            raise LLMError(f"Google AI error: {e}")

    def is_available(self) -> tuple[bool, str]:
        if not self._api_key:
            return False, "GOOGLE_API_KEY not set"
        return True, ""


class OllamaProvider(LLMProvider):
    """Llama 3 via local Ollama server."""

    def __init__(self, model_id: str = "llama3"):
        self._model_id = model_id
        self._base_url = "http://localhost:11434"

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        try:
            import ollama

            response = ollama.chat(
                model=self._model_id,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": max_tokens, "temperature": 0.3},
            )
            return response["message"]["content"]
        except ImportError:
            # Fallback to requests
            return self._generate_via_http(prompt, max_tokens)
        except Exception as e:
            err_str = str(e).lower()
            if "connection" in err_str or "refused" in err_str:
                raise LLMConnectionError(
                    "Ollama server not running. Start with: ollama serve"
                )
            if "not found" in err_str or "pull" in err_str:
                raise LLMError(
                    f"Model '{self._model_id}' not found. "
                    f"Pull it with: ollama pull {self._model_id}"
                )
            raise LLMError(f"Ollama error: {e}")

    def _generate_via_http(self, prompt: str, max_tokens: int) -> str:
        """Fallback HTTP request to Ollama API."""
        import requests

        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {"num_predict": max_tokens, "temperature": 0.3},
                    "stream": False,
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.ConnectionError:
            raise LLMConnectionError(
                "Ollama server not running. Start with: ollama serve"
            )
        except Exception as e:
            raise LLMError(f"Ollama HTTP error: {e}")

    def is_available(self) -> tuple[bool, str]:
        try:
            import requests

            resp = requests.get(f"{self._base_url}/api/tags", timeout=3)
            if resp.status_code != 200:
                return False, "Ollama server not responding"
            models = [m["name"] for m in resp.json().get("models", [])]
            # Check for model name with or without :latest tag
            if any(self._model_id in m for m in models):
                return True, ""
            return False, f"Model '{self._model_id}' not pulled. Run: ollama pull {self._model_id}"
        except Exception:
            return False, "Ollama server not running"


def get_provider(model_name: str) -> LLMProvider:
    """Factory function: given a display name, return the appropriate LLMProvider."""
    model_config = LLM_MODELS.get(model_name)
    if model_config is None:
        raise LLMError(f"Unknown model: {model_name}")

    provider_type = model_config["provider"]
    model_id = model_config["model_id"]
    env_key = model_config.get("env_key")

    api_key = get_api_key(env_key) if env_key else ""

    if provider_type == "openai":
        return OpenAIProvider(api_key=api_key, model_id=model_id)
    elif provider_type == "anthropic":
        return AnthropicProvider(api_key=api_key, model_id=model_id)
    elif provider_type == "google":
        return GoogleProvider(api_key=api_key, model_id=model_id)
    elif provider_type == "ollama":
        return OllamaProvider(model_id=model_id)
    else:
        raise LLMError(f"Unknown provider type: {provider_type}")
