"""Azure AI Foundry LLM interface with dynamic model discovery."""

from typing import Optional
import requests
import json

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
    """Azure AI Foundry provider using direct REST API."""

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

        if not self._endpoint:
            raise LLMAuthError("AZURE_AI_ENDPOINT not set")
        if not USE_AZURE_CREDENTIAL and not self._api_key:
            raise LLMAuthError("AZURE_AI_KEY not set")

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        """Generate text using the Azure AI model via REST API."""
        try:
            # Construct the inference URL with API version
            # Azure AI Foundry format: {endpoint}/models/{model}/chat/completions?api-version=2024-05-01-preview
            url = f"{self._endpoint.rstrip('/')}/models/{self._model_id}/chat/completions"

            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "api-key": self._api_key,
            }

            # Prepare query parameters
            params = {
                "api-version": "2024-05-01-preview"
            }

            # Prepare request body
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful API recommendation assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            # Make the request
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
                raise LLMError(f"Azure AI error (status {response.status_code}): {error_detail}")

            # Parse response
            response.raise_for_status()
            result = response.json()

            # Extract the generated text
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0].get("message", {})
                return message.get("content", "")
            else:
                raise LLMError(f"Unexpected response format: {result}")

        except requests.exceptions.Timeout:
            raise LLMConnectionError("Request to Azure AI timed out")
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(f"Failed to connect to Azure AI: {e}")
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Azure AI generation error: {e}")

    def is_available(self) -> tuple[bool, str]:
        """Check if Azure AI is configured and reachable."""
        if not self._endpoint:
            return False, "AZURE_AI_ENDPOINT not set"
        if not USE_AZURE_CREDENTIAL and not self._api_key:
            return False, "AZURE_AI_KEY not set"
        return True, ""


def fetch_available_models(endpoint: Optional[str] = None, api_key: Optional[str] = None) -> list[dict]:
    """Fetch available models from Azure AI Foundry.

    Returns:
        List of dicts with 'model_id' and 'display_name' keys
    """
    endpoint = endpoint or get_azure_endpoint()
    api_key = api_key or get_azure_key()

    if not endpoint:
        return []

    # Try to fetch models from the Azure AI endpoint
    # Azure AI Foundry format: {endpoint}/models?api-version=2024-05-01-preview
    try:
        url = f"{endpoint.rstrip('/')}/models"
        headers = {
            "api-key": api_key,
        }
        params = {
            "api-version": "2024-05-01-preview"
        }

        response = requests.get(url, headers=headers, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            models = []

            # Parse the models from the response
            # The exact format depends on your Azure AI Foundry deployment
            if isinstance(data, dict) and "data" in data:
                for model in data["data"]:
                    model_id = model.get("id") or model.get("name")
                    display_name = model.get("display_name") or model.get("friendly_name") or model_id
                    if model_id:
                        models.append({
                            "model_id": model_id,
                            "display_name": display_name
                        })
            elif isinstance(data, list):
                for model in data:
                    model_id = model.get("id") or model.get("name")
                    display_name = model.get("display_name") or model.get("friendly_name") or model_id
                    if model_id:
                        models.append({
                            "model_id": model_id,
                            "display_name": display_name
                        })

            if models:
                return models

    except Exception as e:
        print(f"Warning: Could not fetch models from Azure AI Foundry: {e}")

    # Fallback: Return a default list of common models
    # These are typical models available in Azure AI Foundry
    return [
        {"model_id": "gpt-4", "display_name": "GPT-4"},
        {"model_id": "gpt-4-turbo", "display_name": "GPT-4 Turbo"},
        {"model_id": "gpt-4o", "display_name": "GPT-4o"},
        {"model_id": "gpt-4o-mini", "display_name": "GPT-4o Mini"},
        {"model_id": "gpt-35-turbo", "display_name": "GPT-3.5 Turbo"},
        {"model_id": "gpt-35-turbo-16k", "display_name": "GPT-3.5 Turbo 16K"},
    ]


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
        return False, "AZURE_AI_KEY not set"

    return True, ""
