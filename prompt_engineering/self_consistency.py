"""
Self-Consistency Agent

Implements Self-Consistency decoding for improved accuracy through
majority voting across multiple reasoning paths.

References:
- Wang et al., 2022: "Self-Consistency Improves Chain of Thought Reasoning"
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional
import logging

from .base_agent import BasePromptAgent, AgentResult, LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ReasoningPath:
    """A single reasoning path with its answer."""
    reasoning: str
    answer: str
    confidence: float = 1.0


class SelfConsistencyAgent(BasePromptAgent):
    """
    Self-Consistency Agent.
    
    Generates multiple reasoning paths using Chain-of-Thought and
    selects the most consistent answer through majority voting.
    
    This approach trades compute for accuracy - useful when you need
    high confidence in complex reasoning tasks.
    
    Example:
        agent = SelfConsistencyAgent(llm_client=my_llm, num_samples=5)
        result = agent.solve("If 3x + 5 = 20, what is x?")
        print(result.final_answer)  # Most common answer across paths
        print(result.metadata["confidence"])  # Voting confidence
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        num_samples: int = 5,
        temperature: float = 0.7,
        normalize_answers: bool = True,
        **kwargs
    ):
        """
        Initialize Self-Consistency agent.
        
        Args:
            llm_client: LLM client for generation
            num_samples: Number of reasoning paths to generate
            temperature: Temperature for diverse sampling (higher = more diverse)
            normalize_answers: Whether to normalize answers for comparison
            **kwargs: Additional base agent arguments
        """
        super().__init__(llm_client, temperature=temperature, **kwargs)
        self.num_samples = num_samples
        self.normalize_answers = normalize_answers
        
    def _build_cot_prompt(self, problem: str) -> str:
        """Build a Chain-of-Thought prompt."""
        return f"""Solve this problem step by step.

Problem: {problem}

Let's think through this carefully:
1. First, I'll identify what we know
2. Then, I'll work through the solution
3. Finally, I'll state the answer

Solution:
"""
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison."""
        if not self.normalize_answers:
            return answer
        
        # Convert to lowercase
        normalized = answer.lower().strip()
        
        # Remove common punctuation
        normalized = normalized.rstrip(".,!?;:")
        
        # Remove common prefixes
        prefixes_to_remove = [
            "the answer is",
            "answer:",
            "therefore,",
            "so,",
            "thus,",
            "=",
        ]
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        
        # Try to extract just the number if it's a numeric answer
        words = normalized.split()
        for word in words:
            # Check if word is a number
            try:
                float(word.replace(",", ""))
                return word
            except ValueError:
                continue
        
        return normalized
    
    def _generate_path(self, problem: str, **kwargs) -> Optional[ReasoningPath]:
        """Generate a single reasoning path."""
        try:
            prompt = self._build_cot_prompt(problem)
            response = self._call_llm(prompt, **kwargs)
            
            # Extract answer
            answer = self._extract_answer(response)
            
            if answer:
                return ReasoningPath(
                    reasoning=response,
                    answer=answer
                )
            return None
            
        except Exception as e:
            logger.warning(f"Failed to generate path: {e}")
            return None
    
    def solve(
        self,
        problem: str,
        **kwargs
    ) -> AgentResult:
        """
        Solve a problem using self-consistency.
        
        Args:
            problem: The problem to solve
            **kwargs: Additional LLM arguments
            
        Returns:
            AgentResult with voted answer and confidence
        """
        paths: List[ReasoningPath] = []
        
        try:
            # Generate multiple paths
            for i in range(self.num_samples):
                if self.verbose:
                    logger.info(f"Generating path {i + 1}/{self.num_samples}")
                
                path = self._generate_path(problem, **kwargs)
                if path:
                    paths.append(path)
            
            if not paths:
                return AgentResult(
                    success=False,
                    error="No valid reasoning paths generated"
                )
            
            # Normalize and count answers
            normalized_answers = [
                self._normalize_answer(p.answer) for p in paths
            ]
            answer_counts = Counter(normalized_answers)
            
            # Get most common answer
            most_common = answer_counts.most_common(1)[0]
            voted_answer, vote_count = most_common
            
            # Find the full answer (unnormalized) from best path
            best_path_idx = normalized_answers.index(voted_answer)
            full_answer = paths[best_path_idx].answer
            
            # Calculate confidence
            confidence = vote_count / len(paths)
            
            return AgentResult(
                success=True,
                final_answer=full_answer,
                reasoning_steps=[p.reasoning for p in paths],
                metadata={
                    "num_paths": len(paths),
                    "vote_count": vote_count,
                    "confidence": confidence,
                    "all_answers": [p.answer for p in paths],
                    "normalized_answers": normalized_answers,
                    "answer_distribution": dict(answer_counts)
                }
            )
            
        except Exception as e:
            logger.error(f"Self-consistency solve failed: {e}")
            return AgentResult(
                success=False,
                error=str(e)
            )
    
    def solve_with_threshold(
        self,
        problem: str,
        confidence_threshold: float = 0.6,
        max_attempts: int = 3,
        **kwargs
    ) -> AgentResult:
        """
        Solve with a minimum confidence threshold.
        
        If confidence is below threshold, generates more paths
        until threshold is met or max_attempts reached.
        
        Args:
            problem: The problem to solve
            confidence_threshold: Minimum acceptable confidence
            max_attempts: Maximum rounds of sampling
            **kwargs: Additional arguments
            
        Returns:
            AgentResult with answer meeting confidence threshold
        """
        all_paths: List[ReasoningPath] = []
        
        for attempt in range(max_attempts):
            # Generate more paths
            for _ in range(self.num_samples):
                path = self._generate_path(problem, **kwargs)
                if path:
                    all_paths.append(path)
            
            # Check confidence
            if all_paths:
                normalized = [self._normalize_answer(p.answer) for p in all_paths]
                counts = Counter(normalized)
                _, top_count = counts.most_common(1)[0]
                confidence = top_count / len(all_paths)
                
                if confidence >= confidence_threshold:
                    break
        
        # Build final result from all paths
        if not all_paths:
            return AgentResult(success=False, error="No paths generated")
        
        normalized = [self._normalize_answer(p.answer) for p in all_paths]
        counts = Counter(normalized)
        voted, vote_count = counts.most_common(1)[0]
        
        best_idx = normalized.index(voted)
        
        return AgentResult(
            success=True,
            final_answer=all_paths[best_idx].answer,
            reasoning_steps=[p.reasoning for p in all_paths],
            metadata={
                "num_paths": len(all_paths),
                "attempts": attempt + 1,
                "vote_count": vote_count,
                "confidence": vote_count / len(all_paths),
                "answer_distribution": dict(counts)
            }
        )
