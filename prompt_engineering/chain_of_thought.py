"""
Chain-of-Thought (CoT) Prompting Agent

Breaks down complex problems into step-by-step reasoning.
Uses LLM to generate intermediate reasoning steps before final answer.

References:
- Wei et al., 2022: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol
import logging
import os

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for LLM clients - implement this for your LLM provider."""
    async def generate(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        ...


@dataclass
class CoTRequest:
    """Request for Chain-of-Thought reasoning."""
    
    query: str
    context: Optional[str] = None
    num_steps: int = 3
    use_zero_shot: bool = True  # Zero-shot CoT vs Few-shot CoT
    examples: List[Dict] = field(default_factory=list)  # For few-shot


@dataclass
class CoTStep:
    """A single reasoning step."""
    
    step_num: int
    thought: str
    intermediate_result: Optional[str] = None


@dataclass
class CoTResult:
    """Result from Chain-of-Thought reasoning."""
    
    query: str
    reasoning_steps: List[CoTStep]
    final_answer: str
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)


class ChainOfThoughtAgent:
    """
    Universal Chain-of-Thought (CoT) prompting agent.
    
    Supports:
    - Zero-Shot CoT: "Let's think step by step"
    - Few-Shot CoT: Provide examples with reasoning
    - Multi-step reasoning with intermediate outputs
    - Self-consistency (multiple paths + voting)
    
    Example:
        agent = ChainOfThoughtAgent(llm_client=my_llm)
        result = await agent.execute(CoTRequest(
            query="If a train travels 120 miles in 2 hours, what is its speed?"
        ))
        print(result.final_answer)
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        default_model: str = "gpt-4",
        reasoning_model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize CoT agent.
        
        Args:
            llm_client: LLM client implementing the LLMClient protocol
            default_model: Model for main reasoning
            reasoning_model: Model for answer extraction
        """
        self.llm_client = llm_client
        self.default_model = default_model
        self.reasoning_model = reasoning_model
        logger.info("Chain-of-Thought Agent initialized")

    async def execute(self, request: CoTRequest) -> CoTResult:
        """
        Execute Chain-of-Thought reasoning.
        
        Args:
            request: CoTRequest
            
        Returns:
            CoTResult with reasoning steps and final answer
        """
        if request.use_zero_shot:
            return await self._zero_shot_cot(request)
        else:
            return await self._few_shot_cot(request)

    async def _zero_shot_cot(self, request: CoTRequest) -> CoTResult:
        """Zero-shot CoT: Let's think step by step."""
        
        # Build zero-shot CoT prompt
        prompt = f"""Question: {request.query}

{f"Context: {request.context}" if request.context else ""}

Let's think step by step to solve this problem. Break it down into {request.num_steps} clear reasoning steps.

Step 1:"""

        try:
            # Use LLM to generate reasoning
            response = await self._call_llm(
                prompt=prompt,
                model=self.default_model,
                temperature=0.7,
                max_tokens=1000,
            )
            
            # Parse reasoning steps
            reasoning_steps = self._parse_reasoning_steps(response)
            
            # Extract final answer
            final_answer = await self._extract_final_answer(
                request.query, reasoning_steps
            )
            
            logger.info(f"Zero-shot CoT completed: {len(reasoning_steps)} steps")
            
            return CoTResult(
                query=request.query,
                reasoning_steps=reasoning_steps,
                final_answer=final_answer,
                confidence=0.8,
                metadata={"method": "zero_shot_cot", "num_steps": len(reasoning_steps)},
            )
            
        except Exception as e:
            logger.error(f"Zero-shot CoT failed: {e}")
            return CoTResult(
                query=request.query,
                reasoning_steps=[],
                final_answer=f"Error: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)},
            )

    async def _few_shot_cot(self, request: CoTRequest) -> CoTResult:
        """Few-shot CoT: Provide examples with reasoning."""
        
        # Build few-shot CoT prompt with examples
        examples_text = ""
        for i, example in enumerate(request.examples, 1):
            examples_text += f"""
Example {i}:
Question: {example.get('question', '')}
Reasoning:
{example.get('reasoning', '')}
Answer: {example.get('answer', '')}

"""
        
        prompt = f"""{examples_text}

Now solve this question using the same step-by-step reasoning approach:

Question: {request.query}

{f"Context: {request.context}" if request.context else ""}

Reasoning:
Step 1:"""

        try:
            response = await self._call_llm(
                prompt=prompt,
                model=self.default_model,
                temperature=0.7,
                max_tokens=1000,
            )
            
            reasoning_steps = self._parse_reasoning_steps(response)
            final_answer = await self._extract_final_answer(request.query, reasoning_steps)
            
            logger.info(f"Few-shot CoT completed: {len(reasoning_steps)} steps")
            
            return CoTResult(
                query=request.query,
                reasoning_steps=reasoning_steps,
                final_answer=final_answer,
                confidence=0.85,
                metadata={
                    "method": "few_shot_cot",
                    "num_examples": len(request.examples),
                },
            )
            
        except Exception as e:
            logger.error(f"Few-shot CoT failed: {e}")
            return CoTResult(
                query=request.query,
                reasoning_steps=[],
                final_answer=f"Error: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)},
            )

    async def _call_llm(
        self, prompt: str, model: str, temperature: float, max_tokens: int
    ) -> str:
        """
        Call LLM via the configured client.
        
        Override this method to add custom fallback logic.
        """
        if not self.llm_client:
            raise ValueError("No LLM client configured")
        
        response = await self.llm_client.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content if hasattr(response, 'content') else str(response)

    def _parse_reasoning_steps(self, response: str) -> List[CoTStep]:
        """Parse reasoning steps from LLM response."""
        steps = []
        lines = response.strip().split('\n')
        
        current_step = None
        current_thought = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("Step "):
                # Save previous step
                if current_step is not None and current_thought:
                    steps.append(CoTStep(
                        step_num=current_step,
                        thought=' '.join(current_thought),
                    ))
                
                # Start new step
                try:
                    current_step = int(line.split()[1].rstrip(':'))
                    current_thought = [line.split(':', 1)[1].strip()] if ':' in line else []
                except (IndexError, ValueError):
                    current_thought.append(line)
            elif line and current_step is not None:
                current_thought.append(line)
        
        # Save last step
        if current_step is not None and current_thought:
            steps.append(CoTStep(
                step_num=current_step,
                thought=' '.join(current_thought),
            ))
        
        return steps

    async def _extract_final_answer(self, query: str, steps: List[CoTStep]) -> str:
        """Extract final answer from reasoning steps."""
        if not steps:
            return "No reasoning steps generated"
        
        # Build summary of reasoning
        reasoning_summary = "\n".join([
            f"Step {step.step_num}: {step.thought}"
            for step in steps
        ])
        
        # Ask LLM to synthesize final answer
        prompt = f"""Based on this step-by-step reasoning, provide a concise final answer to the question.

Question: {query}

Reasoning:
{reasoning_summary}

Final Answer:"""

        try:
            response = await self._call_llm(
                prompt=prompt,
                model=self.reasoning_model,
                temperature=0.3,
                max_tokens=200,
            )
            return response.strip()
        except Exception as e:
            logger.warning(f"Failed to extract final answer: {e}")
            return steps[-1].thought if steps else "Unable to determine answer"

    async def self_consistency(
        self, request: CoTRequest, num_samples: int = 3
    ) -> CoTResult:
        """
        Self-consistency: Generate multiple reasoning paths and vote.
        
        This technique improves accuracy by generating multiple independent
        reasoning chains and selecting the most common answer.
        
        Args:
            request: CoTRequest
            num_samples: Number of reasoning paths to generate
            
        Returns:
            CoTResult with majority-voted answer
        """
        logger.info(f"Running self-consistency with {num_samples} samples")
        
        # Generate multiple reasoning paths
        results = []
        for i in range(num_samples):
            result = await self.execute(request)
            results.append(result)
        
        # Vote on final answer
        answers = [r.final_answer for r in results]
        final_answer = max(set(answers), key=answers.count)
        
        # Find best reasoning path for this answer
        best_result = next(r for r in results if r.final_answer == final_answer)
        
        # Calculate confidence based on agreement
        agreement_count = answers.count(final_answer)
        confidence = agreement_count / num_samples
        
        logger.info(f"Self-consistency: {agreement_count}/{num_samples} agreement")
        
        return CoTResult(
            query=request.query,
            reasoning_steps=best_result.reasoning_steps,
            final_answer=final_answer,
            confidence=confidence,
            metadata={
                "method": "self_consistency",
                "num_samples": num_samples,
                "agreement": agreement_count,
                "all_answers": answers,
            },
        )
