"""
Agent module - AI-powered analysis and strategy generation.

The Agent is the "Architect" that:
- Analyzes ContextPackets
- Generates StrategyDefinitions (SDL)
- Produces TuningProposals
- Handles error recovery recommendations
"""

from .client import GeminiAgent
from .prompts import PromptBuilder
from .parser import SDLParser

__all__ = [
    "GeminiAgent",
    "PromptBuilder",
    "SDLParser",
]
