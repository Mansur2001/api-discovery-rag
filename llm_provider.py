"""LangChain-based LLM interface for Azure AI Foundry with dynamic model discovery."""

from typing import Optional
import requests

from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel

from config import get_azure_endpoint, get_azure_key, USE_AZURE_CREDENTIAL


class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class LLMAuthError(LLMError):
    """Invalid or missing credentials."""
    pass


class LLMConnectionError(LLMError):
    """Network or API connectivity errors."""
    pass


class LLMRateLimitError(LLMError):
    """Rate limit exceeded."""
    pass


class AzureAIFoundryProvider:
    """Azure AI Foundry provider using LangChain and azure-ai-inference."""

    def __init__(self, model_id: str, endpoint: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize Azure AI Foundry provider.

        Args:
            model_id: The model deployment name from Azure AI Foundry
            endpoint: Azure AI endpoint URL (uses config if not provided)
            api_key: Azure AI key (uses config if not provided)
        """
        self._model_id = model_id
        self._endpoint = endpoint or get_azure_endpoint()
        self._api_key = api_key or get_azure_key()
        self._client: Optional[BaseChatModel] = None

        if not self._endpoint:
            raise LLMAuthError("AZURE_AI_ENDPOINT not set")

        # Initialize the LangChain client
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Azure AI Inference client with LangChain."""
        try:
            from azure.ai.inference import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential
            from azure.identity import DefaultAzureCredential
            from langchain_core.language_models.chat_models import SimpleChatModel

            # Choose authentication method
            if USE_AZURE_CREDENTIAL:
                credential = DefaultAzureCredential()
            else:
                if not self._api_key:
                    raise LLMAuthError("AZURE_AI_KEY not set and USE_AZURE_CREDENTIAL is false")
                credential = AzureKeyCredential(self._api_key)

            # Create Azure AI client
            azure_client = ChatCompletionsClient(
                endpoint=self._endpoint,
                credential=credential
            )

            # Wrap in a simple LangChain-compatible interface
            self._azure_client = azure_client

        except ImportError as e:
            raise LLMError(
                f"Missing required packages: {e}. "
                "Run: pip install azure-ai-inference azure-identity langchain"
            )
        except Exception as e:
            raise LLMError(f"Failed to initialize Azure AI client: {e}")

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        """Generate text using the Azure AI model."""
        try:
            from azure.ai.inference.models import SystemMessage, UserMessage

            response = self._azure_client.complete(
                messages=[
                    SystemMessage(content="You are a helpful API recommendation assistant."),
                    UserMessage(content=prompt)
                ],
                model=self._model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            err_str = str(e).lower()
            if "unauthorized" in err_str or "authentication" in err_str or "credential" in err_str:
                raise LLMAuthError(f"Azure authentication failed: {e}")
            if "quota" in err_str or "rate limit" in err_str or "429" in err_str:
                raise LLMRateLimitError(f"Azure rate limit exceeded: {e}")
            if "connection" in err_str or "network" in err_str or "timeout" in err_str:
                raise LLMConnectionError(f"Azure connection error: {e}")
            raise LLMError(f"Azure AI generation error: {e}")

    def is_available(self) -> tuple[bool, str]:
        """Check if Azure AI is configured and reachable."""
        if not self._endpoint:
            return False, "AZURE_AI_ENDPOINT not set"
        if not USE_AZURE_CREDENTIAL and not self._api_key:
            return False, "AZURE_AI_KEY not set (or enable USE_AZURE_CREDENTIAL)"

        # Try a test request to verify connectivity
        try:
            # Quick health check - just verify the client initializes
            if self._azure_client is None:
                return False, "Azure client not initialized"
            return True, ""
        except Exception as e:
            return False, f"Connection test failed: {e}"


def fetch_available_models(endpoint: Optional[str] = None, api_key: Optional[str] = None) -> list[dict]:
    """Fetch available models from Azure AI Foundry Model Catalog.

    Returns:
        List of dicts with 'model_id' and 'display_name' keys
    """
    endpoint = endpoint or get_azure_endpoint()
    api_key = api_key or get_azure_key()

    if not endpoint:
        return []

    # Azure AI Foundry model catalog endpoint
    # The exact endpoint depends on your deployment
    # This is a generic approach - you may need to adjust based on your Foundry setup

    try:
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential
        from azure.identity import DefaultAzureCredential

        # Try to get model info via the inference API
        # Note: The model catalog API varies by deployment
        # For now, we'll return a default set of common models
        # You can customize this list based on your Foundry deployment

        models = [
            {"model_id": "gpt-4", "display_name": "GPT-4"},
            {"model_id": "gpt-4-turbo", "display_name": "GPT-4 Turbo"},
            {"model_id": "gpt-4o", "display_name": "GPT-4o"},
            {"model_id": "gpt-35-turbo", "display_name": "GPT-3.5 Turbo"},
            {"model_id": "claude-3-5-sonnet", "display_name": "Claude 3.5 Sonnet"},
            {"model_id": "claude-3-opus", "display_name": "Claude 3 Opus"},
            {"model_id": "claude-3-haiku", "display_name": "Claude 3 Haiku"},
            {"model_id": "gemini-1.5-pro", "display_name": "Gemini 1.5 Pro"},
            {"model_id": "gemini-1.5-flash", "display_name": "Gemini 1.5 Flash"},
            {"model_id": "llama-3-70b", "display_name": "Llama 3 70B"},
            {"model_id": "llama-3-8b", "display_name": "Llama 3 8B"},
            {"model_id": "mistral-large", "display_name": "Mistral Large"},
            {"model_id": "mistral-small", "display_name": "Mistral Small"},
        ]

        # TODO: If your Azure AI Foundry has a model catalog API, fetch from there
        # Example (adjust based on your actual API):
        # if USE_AZURE_CREDENTIAL:
        #     credential = DefaultAzureCredential()
        # else:
        #     credential = AzureKeyCredential(api_key)
        #
        # catalog_endpoint = f"{endpoint}/models"  # Adjust based on your API
        # response = requests.get(catalog_endpoint, headers={"Authorization": f"Bearer {api_key}"})
        # if response.status_code == 200:
        #     models = response.json().get("models", [])

        return models

    except Exception as e:
        # If model fetching fails, return empty list
        # The UI will show a warning
        print(f"Warning: Could not fetch models from Azure AI Foundry: {e}")
        return []


def get_provider(model_id: str) -> AzureAIFoundryProvider:
    """Factory function: create Azure AI Foundry provider for the given model."""
    return AzureAIFoundryProvider(model_id=model_id)


def check_azure_availability() -> tuple[bool, str]:
    """Check if Azure AI Foundry is configured."""
    endpoint = get_azure_endpoint()
    api_key = get_azure_key()

    if not endpoint:
        return False, "AZURE_AI_ENDPOINT not set"
    if not USE_AZURE_CREDENTIAL and not api_key:
        return False, "AZURE_AI_KEY not set (or enable USE_AZURE_CREDENTIAL)"

    return True, ""
