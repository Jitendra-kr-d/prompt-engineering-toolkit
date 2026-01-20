"""
Tree of Thoughts (ToT) Agent

Structures reasoning as a branching tree where models explore multiple paths
simultaneously, evaluate intermediate steps, and backtrack when needed.

References:
- Yao et al., 2023: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Protocol
import logging

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for LLM clients."""
    async def generate(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        ...


@dataclass
class ThoughtNode:
    """A node in the tree of thoughts."""
    
    id: str
    thought: str
    depth: int
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    value_score: float = 0.0  # Evaluation score
    is_leaf: bool = False
    is_solution: bool = False


@dataclass
class ToTRequest:
    """Request for Tree of Thoughts reasoning."""
    
    query: str
    context: Optional[str] = None
    max_depth: int = 3
    branches_per_node: int = 3
    evaluation_strategy: str = "value"  # "value" or "vote"
    prune_threshold: float = 0.5  # Minimum score to continue exploration


@dataclass
class ToTResult:
    """Result from Tree of Thoughts reasoning."""
    
    query: str
    best_path: List[ThoughtNode]
    all_nodes: List[ThoughtNode]
    final_answer: str
    confidence: float
    metadata: Dict = field(default_factory=dict)


class TreeOfThoughtsAgent:
    """
    Universal Tree of Thoughts (ToT) agent.
    
    Features:
    - Branching exploration of reasoning paths
    - Intermediate step evaluation
    - Backtracking when path is unproductive
    - Parallel path exploration
    
    Example:
        agent = TreeOfThoughtsAgent(llm_client=my_llm)
        result = await agent.execute(ToTRequest(
            query="Make 24 using [4, 7, 8, 8]",
            max_depth=3
        ))
        print(result.final_answer)
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        generator_model: str = "gpt-4",
        evaluator_model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize ToT agent.
        
        Args:
            llm_client: LLM client for generation
            generator_model: Model for generating thoughts
            evaluator_model: Model for evaluating thoughts
        """
        self.llm_client = llm_client
        self.generator_model = generator_model
        self.evaluator_model = evaluator_model
        self.nodes: Dict[str, ThoughtNode] = {}
        logger.info("Tree of Thoughts Agent initialized")

    async def execute(self, request: ToTRequest) -> ToTResult:
        """
        Execute Tree of Thoughts reasoning.
        
        Args:
            request: ToTRequest
            
        Returns:
            ToTResult with best reasoning path
        """
        logger.info(f"Starting ToT: depth={request.max_depth}, branches={request.branches_per_node}")
        
        # Reset nodes for new execution
        self.nodes = {}
        
        # Initialize root node
        root_id = "root"
        root_node = ThoughtNode(
            id=root_id,
            thought=f"Initial state: {request.query}",
            depth=0,
            value_score=1.0,
        )
        self.nodes[root_id] = root_node
        
        # Build tree level by level
        await self._build_tree(request, root_id)
        
        # Find best path
        best_path = self._find_best_path(request)
        
        # Generate final answer from best path
        final_answer = await self._synthesize_answer(request, best_path)
        
        # Calculate confidence based on path quality
        confidence = self._calculate_confidence(best_path)
        
        logger.info(f"ToT completed: {len(self.nodes)} nodes, confidence={confidence:.2f}")
        
        return ToTResult(
            query=request.query,
            best_path=best_path,
            all_nodes=list(self.nodes.values()),
            final_answer=final_answer,
            confidence=confidence,
            metadata={
                "total_nodes": len(self.nodes),
                "max_depth_reached": max(n.depth for n in self.nodes.values()),
            },
        )

    async def _build_tree(self, request: ToTRequest, node_id: str):
        """Recursively build the tree of thoughts."""
        node = self.nodes[node_id]
        
        # Stop if max depth reached
        if node.depth >= request.max_depth:
            node.is_leaf = True
            return
        
        # Generate branches from this node
        branches = await self._generate_branches(request, node)
        
        # Evaluate all branches
        evaluated_branches = await self._evaluate_branches(request, branches)
        
        # Select top branches based on scores
        top_branches = sorted(evaluated_branches, key=lambda x: x[1], reverse=True)[
            :request.branches_per_node
        ]
        
        # Create child nodes for top branches
        for branch_text, score in top_branches:
            child_id = f"{node_id}_child_{len(node.children_ids)}"
            child_node = ThoughtNode(
                id=child_id,
                thought=branch_text,
                depth=node.depth + 1,
                parent_id=node_id,
                value_score=score,
            )
            self.nodes[child_id] = child_node
            node.children_ids.append(child_id)
            
            # Recursively expand promising children (pruning)
            if score > request.prune_threshold:
                await self._build_tree(request, child_id)
            else:
                child_node.is_leaf = True

    async def _generate_branches(
        self, request: ToTRequest, node: ThoughtNode
    ) -> List[str]:
        """Generate possible next thoughts from current node."""
        
        # Build context from path so far
        path = self._get_path_to_node(node.id)
        context = "\n".join([f"Step {i+1}: {n.thought}" for i, n in enumerate(path)])
        
        prompt = f"""We are solving this problem: {request.query}

Current reasoning path:
{context}

Generate {request.branches_per_node} different possible next reasoning steps.
Each step should explore a different approach or aspect.

Provide them as a numbered list:
1."""

        try:
            response = await self._call_llm(
                prompt=prompt,
                model=self.generator_model,
                temperature=0.9,  # Higher temperature for diversity
                max_tokens=500,
            )
            
            # Parse branches
            branches = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering
                    branch = line.split('.', 1)[-1].strip()
                    if branch:
                        branches.append(branch)
            
            return branches[:request.branches_per_node]
            
        except Exception as e:
            logger.error(f"Branch generation failed: {e}")
            return [f"Continue exploring: {node.thought}"]

    async def _evaluate_branches(
        self, request: ToTRequest, branches: List[str]
    ) -> List[Tuple[str, float]]:
        """Evaluate all branches and return with scores."""
        evaluated = []
        
        # Evaluate in parallel for efficiency
        tasks = [self._evaluate_branch(request, branch) for branch in branches]
        scores = await asyncio.gather(*tasks, return_exceptions=True)
        
        for branch, score in zip(branches, scores):
            if isinstance(score, Exception):
                logger.warning(f"Evaluation failed for branch: {score}")
                score = 0.5  # Default neutral score
            evaluated.append((branch, score))
        
        return evaluated

    async def _evaluate_branch(
        self, request: ToTRequest, branch: str
    ) -> float:
        """Evaluate how promising a branch is (0.0-1.0)."""
        
        prompt = f"""Evaluate how promising this reasoning step is for solving the problem.

Problem: {request.query}
Reasoning step: {branch}

Score this step on a scale of 0.0 to 1.0 where:
- 1.0 = Highly promising, directly advances toward solution
- 0.5 = Neutral, might be useful
- 0.0 = Unpromising, unlikely to help

Provide only the numerical score:"""

        try:
            response = await self._call_llm(
                prompt=prompt,
                model=self.evaluator_model,
                temperature=0.3,
                max_tokens=50,
            )
            
            # Extract score
            score_text = response.strip().split()[0]
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Branch evaluation failed: {e}")
            return 0.5  # Default neutral score

    def _find_best_path(self, request: ToTRequest) -> List[ThoughtNode]:
        """Find the best path from root to a leaf."""
        
        # Find all leaf nodes
        leaf_nodes = [n for n in self.nodes.values() if n.is_leaf or not n.children_ids]
        
        if not leaf_nodes:
            deepest = max(self.nodes.values(), key=lambda n: n.depth)
            return self._get_path_to_node(deepest.id)
        
        # Calculate path scores (average of node values)
        best_path = None
        best_score = -1
        
        for leaf in leaf_nodes:
            path = self._get_path_to_node(leaf.id)
            path_score = sum(n.value_score for n in path) / len(path)
            
            if path_score > best_score:
                best_score = path_score
                best_path = path
        
        return best_path if best_path else [self.nodes["root"]]

    def _get_path_to_node(self, node_id: str) -> List[ThoughtNode]:
        """Get path from root to specified node."""
        path = []
        current_id = node_id
        
        while current_id:
            node = self.nodes.get(current_id)
            if not node:
                break
            path.insert(0, node)
            current_id = node.parent_id
        
        return path

    async def _synthesize_answer(
        self, request: ToTRequest, path: List[ThoughtNode]
    ) -> str:
        """Synthesize final answer from best reasoning path."""
        
        path_text = "\n".join([
            f"Step {i}: {node.thought}"
            for i, node in enumerate(path, 1)
        ])
        
        prompt = f"""Based on this reasoning path, provide a final answer to the question.

Question: {request.query}

Reasoning path:
{path_text}

Final Answer:"""

        try:
            response = await self._call_llm(
                prompt=prompt,
                model=self.evaluator_model,
                temperature=0.3,
                max_tokens=300,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return path[-1].thought if path else "Unable to determine answer"

    def _calculate_confidence(self, path: List[ThoughtNode]) -> float:
        """Calculate confidence based on path quality."""
        if not path:
            return 0.0
        
        # Average of node scores
        avg_score = sum(n.value_score for n in path) / len(path)
        
        # Penalize shallow paths
        depth_factor = min(1.0, len(path) / 3.0)
        
        return avg_score * depth_factor

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
