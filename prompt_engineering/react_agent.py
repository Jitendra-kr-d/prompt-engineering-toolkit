"""
ReAct (Reasoning + Acting) Framework Agent

Combines reasoning and acting by interleaving thought processes with actions.
The agent reasons about what to do, takes action, observes results, and continues.

References:
- Yao et al., 2022: "ReAct: Synergizing Reasoning and Acting in Language Models"
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol
import logging

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for LLM clients."""
    async def generate(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        ...


@dataclass
class ReActStep:
    """A single ReAct step (Thought -> Action -> Observation)."""
    
    step_num: int
    thought: str
    action: str
    action_input: Dict
    observation: str
    

@dataclass
class ReActRequest:
    """Request for ReAct reasoning."""
    
    query: str
    context: Optional[str] = None
    available_tools: List[str] = field(default_factory=list)
    max_steps: int = 5


@dataclass
class ReActResult:
    """Result from ReAct reasoning."""
    
    query: str
    steps: List[ReActStep]
    final_answer: str
    success: bool
    metadata: Dict = field(default_factory=dict)


class ReActAgent:
    """
    Universal ReAct (Reasoning + Acting) agent.
    
    Features:
    - Interleaved reasoning and acting
    - Tool/action execution
    - Observation-based adaptation
    - Multi-step problem solving
    
    Example:
        tools = {
            "search": search_function,
            "calculate": calc_function,
        }
        agent = ReActAgent(llm_client=my_llm, tools=tools)
        result = await agent.execute(ReActRequest(
            query="What is the population of Tokyo?",
            available_tools=["search", "calculate"]
        ))
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        tools: Dict[str, Callable] = None,
        default_model: str = "gpt-4",
        reasoning_model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize ReAct agent.
        
        Args:
            llm_client: LLM client for generation
            tools: Dictionary of available tools/actions (name -> callable)
            default_model: Model for reasoning
            reasoning_model: Model for synthesis
        """
        self.llm_client = llm_client
        self.tools = tools or {}
        self.default_model = default_model
        self.reasoning_model = reasoning_model
        logger.info(f"ReAct Agent initialized with tools: {list(self.tools.keys())}")

    async def execute(self, request: ReActRequest) -> ReActResult:
        """
        Execute ReAct reasoning loop.
        
        Args:
            request: ReActRequest
            
        Returns:
            ReActResult with reasoning steps and final answer
        """
        logger.info(f"Starting ReAct: query='{request.query}', max_steps={request.max_steps}")
        
        steps = []
        context_history = []
        
        for step_num in range(1, request.max_steps + 1):
            # Build prompt with history
            prompt = self._build_react_prompt(
                request, steps, context_history, step_num
            )
            
            # Get thought and action from LLM
            thought, action, action_input = await self._get_next_action(prompt)
            
            # Execute action and get observation
            observation = await self._execute_action(action, action_input)
            
            # Create step
            step = ReActStep(
                step_num=step_num,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
            )
            steps.append(step)
            context_history.append(
                f"Step {step_num}:\nThought: {thought}\nAction: {action}\nObservation: {observation}"
            )
            
            # Check if we have final answer
            if action.lower() == "finish" or "final answer" in thought.lower():
                final_answer = observation
                logger.info(f"ReAct completed in {len(steps)} steps")
                return ReActResult(
                    query=request.query,
                    steps=steps,
                    final_answer=final_answer,
                    success=True,
                    metadata={"num_steps": len(steps)},
                )
        
        # Max steps reached
        final_answer = await self._synthesize_final_answer(request, steps)
        logger.warning(f"ReAct max steps reached: {len(steps)}")
        
        return ReActResult(
            query=request.query,
            steps=steps,
            final_answer=final_answer,
            success=False,
            metadata={"num_steps": len(steps), "max_steps_reached": True},
        )

    def _build_react_prompt(
        self, request: ReActRequest, steps: List[ReActStep], 
        history: List[str], step_num: int
    ) -> str:
        """Build ReAct prompt with history."""
        
        tools_text = "\n".join([f"- {tool}" for tool in request.available_tools])
        history_text = "\n\n".join(history) if history else "No previous steps."
        
        prompt = f"""Answer this question by reasoning and taking actions step-by-step.

Question: {request.query}

{f"Context: {request.context}" if request.context else ""}

Available Actions:
{tools_text}
- Finish[answer] - Provide final answer

Previous Steps:
{history_text}

Step {step_num}:
Thought: What should I do next to answer the question?
"""
        return prompt

    async def _get_next_action(self, prompt: str) -> tuple:
        """Get next thought and action from LLM."""
        
        try:
            response = await self._call_llm(
                prompt=prompt,
                model=self.default_model,
                temperature=0.7,
                max_tokens=500,
            )
            
            # Parse thought, action, action_input
            thought = ""
            action = ""
            action_input = {}
            
            lines = response.strip().split('\n')
            for i, line in enumerate(lines):
                if "Thought:" in line:
                    thought = line.split("Thought:", 1)[1].strip()
                elif "Action:" in line:
                    action_line = line.split("Action:", 1)[1].strip()
                    # Parse action and input
                    if '[' in action_line and ']' in action_line:
                        action = action_line.split('[')[0].strip()
                        action_input_str = action_line.split('[')[1].split(']')[0]
                        action_input = {"query": action_input_str}
                    else:
                        action = action_line
            
            if not thought:
                thought = "Continue exploring the problem"
            if not action:
                action = "Finish"
                action_input = {"answer": "Unable to determine next action"}
            
            return thought, action, action_input
            
        except Exception as e:
            logger.error(f"Failed to get next action: {e}")
            return "Error occurred", "Finish", {"answer": f"Error: {str(e)}"}

    async def _execute_action(self, action: str, action_input: Dict) -> str:
        """Execute action and return observation."""
        
        action_lower = action.lower()
        
        # Handle finish action
        if action_lower == "finish":
            return action_input.get("answer", "")
        
        # Execute tool if available
        if action in self.tools:
            try:
                result = await self.tools[action](**action_input)
                return str(result)
            except Exception as e:
                return f"Error executing {action}: {str(e)}"
        
        return f"Action '{action}' not available. Available: {list(self.tools.keys())}"

    async def _synthesize_final_answer(
        self, request: ReActRequest, steps: List[ReActStep]
    ) -> str:
        """Synthesize final answer from steps."""
        
        steps_text = "\n".join([
            f"Step {s.step_num}: {s.thought}\nAction: {s.action}\nResult: {s.observation}"
            for s in steps
        ])
        
        prompt = f"""Based on these reasoning and action steps, provide a final answer.

Question: {request.query}

Steps taken:
{steps_text}

Final Answer:"""

        try:
            response = await self._call_llm(
                prompt=prompt,
                model=self.reasoning_model,
                temperature=0.3,
                max_tokens=300,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Final answer synthesis failed: {e}")
            return steps[-1].observation if steps else "Unable to determine answer"

    async def _call_llm(
        self, prompt: str, model: str, temperature: float, max_tokens: int
    ) -> str:
        """Call LLM via the configured client."""
        if not self.llm_client:
            raise ValueError("No LLM client configured")
        
        response = await self.llm_client.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content if hasattr(response, 'content') else str(response)
