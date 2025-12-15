from openai import OpenAI
from .config import settings
from typing import Optional


class OpenAIClient:
    """
    A wrapper class for the OpenAI client to manage API interactions.
    """
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.agent_model

    def get_client(self) -> OpenAI:
        """
        Returns the initialized OpenAI client.
        """
        return self.client

    def get_model(self) -> str:
        """
        Returns the configured model name.
        """
        return self.model


# Create a single instance of the OpenAI client
openai_client = OpenAIClient()