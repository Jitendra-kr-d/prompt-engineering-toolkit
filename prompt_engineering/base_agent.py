"""
Base Prompt Agent

Abstract base class for all prompt engineering agents.
Provides common functionality for LLM interaction and response parsing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol
import logging

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for LLM clients - implement this interface for your LLM."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        ...


@dataclass
class AgentResult:
    """Result from a prompt engineering agent."""
    success: bool
    final_answer: Optional[str] = None
    reasoning_steps: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class BasePromptAgent(ABC):
    """
    Abstract base class for prompt engineering agents.
    
    All prompt engineering techniques inherit from this class.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        verbose: bool = False
    ):
        """
        Initialize the base agent.
        
        Args:
            llm_client: LLM client implementing the LLMClient protocol
            temperature: Sampling temperature for LLM
            max_tokens: Maximum tokens in response
            verbose: Enable verbose logging
        """
        self.llm = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        
    def _call_llm(self, prompt: str, **kwargs) -> str:
        """
        Call the LLM with the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments for the LLM
            
        Returns:
            The LLM response as a string
        """
        if self.verbose:
            logger.info(f"Calling LLM with prompt ({len(prompt)} chars)")
            
        try:
            response = self.llm.generate(
                prompt,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens)
            )
            
            if self.verbose:
                logger.info(f"LLM response ({len(response)} chars)")
                
            return response
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _parse_steps(self, response: str) -> List[str]:
        """
        Parse reasoning steps from LLM response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of reasoning steps
        """
        steps = []
        lines = response.strip().split("\n")
        
        current_step = []
        for line in lines:
            line = line.strip()
            if not line:
                if current_step:
                    steps.append(" ".join(current_step))
                    current_step = []
            else:
                # Check for step markers
                if line.startswith(("Step ", "step ", "1.", "2.", "3.", "4.", "5.")):
                    if current_step:
                        steps.append(" ".join(current_step))
                    current_step = [line]
                else:
                    current_step.append(line)
        
        if current_step:
            steps.append(" ".join(current_step))
            
        return steps
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """
        Extract the final answer from a response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Extracted answer or None
        """
        # Look for common answer patterns
        answer_markers = [
            "The answer is",
            "Answer:",
            "Final answer:",
            "Therefore,",
            "So,",
            "Thus,",
            "In conclusion,",
        ]
        
        lines = response.strip().split("\n")
        
        for line in reversed(lines):
            line = line.strip()
            for marker in answer_markers:
                if marker.lower() in line.lower():
                    # Extract everything after the marker
                    idx = line.lower().find(marker.lower())
                    answer = line[idx + len(marker):].strip()
                    # Clean up common punctuation
                    answer = answer.strip(".,!?")
                    if answer:
                        return answer
        
        # If no marker found, return the last non-empty line
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith(("Step", "Thought", "Action")):
                return line
                
        return None
    
    @abstractmethod
    def solve(self, problem: str, **kwargs) -> AgentResult:
        """
        Solve a problem using this prompt engineering technique.
        
        Args:
            problem: The problem to solve
            **kwargs: Technique-specific arguments
            
        Returns:
            AgentResult with solution and reasoning
        """
        pass
