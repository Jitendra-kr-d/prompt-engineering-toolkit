"""
Prompt Engineering Toolkit

Production-ready implementations of advanced prompt engineering techniques.
"""

from .base_agent import BasePromptAgent, AgentResult, LLMClient
from .chain_of_thought import ChainOfThoughtAgent, CoTRequest, CoTResult, CoTStep
from .tree_of_thoughts import TreeOfThoughtsAgent, ToTRequest, ToTResult, ThoughtNode
from .react_agent import ReActAgent, ReActRequest, ReActResult, ReActStep
from .self_consistency import SelfConsistencyAgent

__all__ = [
    # Base
    "BasePromptAgent",
    "AgentResult",
    "LLMClient",
    # Chain of Thought
    "ChainOfThoughtAgent",
    "CoTRequest",
    "CoTResult",
    "CoTStep",
    # Tree of Thoughts
    "TreeOfThoughtsAgent",
    "ToTRequest",
    "ToTResult",
    "ThoughtNode",
    # ReAct
    "ReActAgent",
    "ReActRequest",
    "ReActResult",
    "ReActStep",
    # Self-Consistency
    "SelfConsistencyAgent",
]

__version__ = "1.0.0"
