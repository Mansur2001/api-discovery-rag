"""Azure LLM interface supporting both Azure OpenAI and Azure AI Foundry."""

from typing import Optional
import requests
import json

from config import get_azure_endpoint, get_azure_key, USE_AZURE_CREDENTIAL, OPENAI_API_KEY, USE_OPENAI


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


class AzureProvider:
    """Unified Azure provider supporting both Azure OpenAI and Azure AI Foundry."""

    def __init__(self, model_id: str, endpoint: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize Azure provider.

        Args:
            model_id: The model deployment name
            endpoint: Azure endpoint URL (uses config if not provided)
            api_key: Azure API key (uses config if not provided)
        """
        self._model_id = model_id
        self._endpoint = endpoint or get_azure_endpoint()
        self._api_key = api_key or get_azure_key()

        if not self._endpoint:
            raise LLMAuthError("AZURE_AI_ENDPOINT not set")
        if not USE_AZURE_CREDENTIAL and not self._api_key:
            raise LLMAuthError("AZURE_AI_KEY not set")

        # Detect endpoint type
        self._is_openai_service = "openai.azure.com" in self._endpoint

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        """Generate text using Azure."""
        try:
            if self._is_openai_service:
                return self._generate_openai_service(prompt, max_tokens, temperature)
            else:
                return self._generate_ai_foundry(prompt, max_tokens, temperature)
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Azure generation error: {e}")

    def _generate_openai_service(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using Azure OpenAI Service endpoint."""
        # Azure OpenAI format: https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version=2024-02-15-preview
        url = f"{self._endpoint.rstrip('/')}/openai/deployments/{self._model_id}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "api-key": self._api_key,
        }

        params = {
            "api-version": "2024-02-15-preview"
        }

        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful API recommendation assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        return self._make_request(url, headers, params, payload)

    def _generate_ai_foundry(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using Azure AI Foundry endpoint."""
        # Try multiple endpoint formats for AI Foundry
        formats = [
            f"{self._endpoint.rstrip('/')}/chat/completions",  # Direct chat endpoint
            f"{self._endpoint.rstrip('/')}/models/{self._model_id}/chat/completions",  # Model-specific
        ]

        headers = {
            "Content-Type": "application/json",
            "api-key": self._api_key,
        }

        # Try multiple API versions
        api_versions = ["2024-02-01", "2024-03-01-preview", "2024-05-01-preview"]

        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful API recommendation assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Try each combination
        last_error = None
        for url in formats:
            for version in api_versions:
                try:
                    params = {"api-version": version}
                    return self._make_request(url, headers, params, payload)
                except LLMError as e:
                    last_error = e
                    continue

        # If all attempts failed, raise the last error
        if last_error:
            raise last_error
        raise LLMError("All AI Foundry endpoint formats failed")

    def _make_request(self, url: str, headers: dict, params: dict, payload: dict) -> str:
        """Make HTTP request and parse response."""
        response = requests.post(url, headers=headers, params=params, json=payload, timeout=60)

        # Handle error responses
        if response.status_code == 401:
            raise LLMAuthError(f"Azure authentication failed: {response.text}")
        elif response.status_code == 429:
            raise LLMRateLimitError(f"Azure rate limit exceeded: {response.text}")
        elif response.status_code >= 400:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("error", {}).get("message", error_detail)
            except:
                pass
            raise LLMError(f"Azure error (status {response.status_code}): {error_detail}")

        # Parse successful response
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            return message.get("content", "")
        else:
            raise LLMError(f"Unexpected response format: {result}")

    def is_available(self) -> tuple[bool, str]:
        """Check if Azure is configured."""
        if not self._endpoint:
            return False, "AZURE_AI_ENDPOINT not set"
        if not USE_AZURE_CREDENTIAL and not self._api_key:
            return False, "AZURE_AI_KEY not set"
        return True, ""


def fetch_available_models(endpoint: Optional[str] = None, api_key: Optional[str] = None) -> list[dict]:
    """Fetch available models from Azure.

    Returns:
        List of dicts with 'model_id' and 'display_name' keys
    """
    endpoint = endpoint or get_azure_endpoint()
    api_key = api_key or get_azure_key()

    if not endpoint:
        return []

    is_openai_service = "openai.azure.com" in endpoint

    try:
        if is_openai_service:
            # Azure OpenAI Service: fetch deployments
            url = f"{endpoint.rstrip('/')}/openai/deployments"
            params = {"api-version": "2024-02-15-preview"}
        else:
            # Azure AI Foundry: fetch models
            url = f"{endpoint.rstrip('/')}/models"
            params = {"api-version": "2024-02-01"}

        headers = {"api-key": api_key}
        response = requests.get(url, headers=headers, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            models = []

            # Parse response based on format
            if isinstance(data, dict) and "data" in data:
                for model in data["data"]:
                    model_id = model.get("id") or model.get("model") or model.get("name")
                    display_name = model.get("display_name") or model.get("friendly_name") or model_id
                    if model_id:
                        models.append({"model_id": model_id, "display_name": display_name})
            elif isinstance(data, list):
                for model in data:
                    model_id = model.get("id") or model.get("model") or model.get("name")
                    display_name = model.get("display_name") or model.get("friendly_name") or model_id
                    if model_id:
                        models.append({"model_id": model_id, "display_name": display_name})

            if models:
                return models
    except Exception as e:
        print(f"Warning: Could not fetch models from Azure: {e}")

    # Fallback: Return common deployment names
    return [
        {"model_id": "gpt-35-turbo", "display_name": "GPT-3.5 Turbo"},
        {"model_id": "gpt-4", "display_name": "GPT-4"},
        {"model_id": "gpt-4o", "display_name": "GPT-4o"},
        {"model_id": "gpt-4o-mini", "display_name": "GPT-4o Mini"},
    ]


def get_provider(model_id: str) -> AzureProvider:
    """Factory function: create Azure provider for the given model."""
    return AzureProvider(model_id=model_id)


def check_azure_availability() -> tuple[bool, str]:
    """Check if Azure is configured."""
    endpoint = get_azure_endpoint()
    api_key = get_azure_key()

    if not endpoint:
        return False, "AZURE_AI_ENDPOINT not set"
    if not USE_AZURE_CREDENTIAL and not api_key:
        return False, "AZURE_AI_KEY not set"

    return True, ""
