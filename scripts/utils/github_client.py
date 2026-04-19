"""GitHub API client utilities."""

from dotenv import load_dotenv
from github import Github, Auth
import os


def load_env() -> str:
    """Load environment variables and return GitHub API token."""
    load_dotenv()
    token = os.getenv("GITHUB_API_TOKEN")
    if not token:
        raise ValueError("GITHUB_API_TOKEN not found in environment variables")
    return token


def get_github_client(api_token: str, seconds_between_requests: float = None) -> Github:
    """
    Create and configure a GitHub API client.

    Args:
        api_token: GitHub personal access token
        seconds_between_requests: Rate limiting delay between requests

    Returns:
        Configured PyGithub client instance
    """
    return Github(
        auth=Auth.Token(api_token),
        per_page=100,
        seconds_between_requests=seconds_between_requests,
        lazy=True,
    )
