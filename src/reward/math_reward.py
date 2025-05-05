#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reward functions for mathematical reasoning with Gemma 3.

This module provides JAX-compatible reward functions for evaluating
mathematical reasoning in model-generated responses, specifically designed
for training with Group Relative Policy Optimization (GRPO) on TPUs.
"""

import re
import logging
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

# Special tokens for formatting responses
REASONING_START = "<start_reasoning>"
REASONING_END = "<end_reasoning>"
SOLUTION_START = "<answer>"
SOLUTION_END = "</answer>"

# Compile regex pattern for extracting reasoning and answer
FORMAT_PATTERN = re.compile(
    rf"{REASONING_START}(.*?){REASONING_END}.*?{SOLUTION_START}(.*?){SOLUTION_END}",
    flags=re.DOTALL
)

def extract_reasoning_and_answer(response: str) -> Optional[Dict[str, str]]:
    """
    Extract reasoning and answer from a formatted response.
    
    Args:
        response: Model generated response
        
    Returns:
        Dictionary with reasoning and answer or None if format doesn't match
    """
    match = FORMAT_PATTERN.search(response)
    if match:
        return {
            "reasoning": match.group(1).strip(),
            "answer": match.group(2).strip()
        }
    return None

def format_exact_reward(completions: List[Dict], **kwargs) -> List[float]:
    """
    Reward function for exact format matching.
    
    Args:
        completions: List of model completions
        
    Returns:
        List of scores (higher is better)
    """
    scores = []
    for completion in completions:
        score = 0.0
        response = completion[0]["content"]
        
        # Match if format is seen exactly
        if FORMAT_PATTERN.search(response) is not None:
            score = 3.0
        
        scores.append(score)
    
    return scores

def format_approximate_reward(completions: List[Dict], **kwargs) -> List[float]:
    """
    Reward function for approximate format matching.
    
    Args:
        completions: List of model completions
        
    Returns:
        List of scores (higher is better)
    """
    scores = []
    for completion in completions:
        score = 0.0
        response = completion[0]["content"]
        
        # Count how many keywords are seen - we penalize if too many
        score += 0.5 if response.count(REASONING_START) == 1 else -0.5
        score += 0.5 if response.count(REASONING_END) == 1 else -0.5
        score += 0.5 if response.count(SOLUTION_START) == 1 else -0.5
        score += 0.5 if response.count(SOLUTION_END) == 1 else -0.5
        
        scores.append(score)
    
    return scores

def answer_matching_reward(prompts: List[Dict], completions: List[Dict], answer: List[str], **kwargs) -> List[float]:
    """
    Reward function for answer correctness.
    
    Args:
        prompts: List of input prompts
        completions: List of model completions
        answer: List of ground truth answers
        
    Returns:
        List of scores (higher is better)
    """
    scores = []
    
    for completion, ground_truth in zip(completions, answer):
        score = 0.0
        response = completion[0]["content"]
        
        # Extract reasoning and answer
        extracted = extract_reasoning_and_answer(response)
        if not extracted:
            scores.append(score)
            continue
            
        extracted_answer = extracted["answer"]
        
        # Exact match
        if extracted_answer == ground_truth:
            score = 3.0
        # Match after stripping spaces
        elif extracted_answer.strip() == ground_truth.strip():
            score = 1.5
        else:
            # Try numerical comparison
            try:
                # Clean up and extract numbers
                extracted_num = float(''.join(filter(
                    lambda x: x.isdigit() or x == '.', extracted_answer)))
                true_num = float(''.join(filter(
                    lambda x: x.isdigit() or x == '.', ground_truth)))
                
                # Calculate ratio for partial scoring
                if true_num == 0:
                    # Handle division by zero
                    score = -1.0 if extracted_num != 0 else 0.5
                else:
                    ratio = extracted_num / true_num
                    if 0.9 <= ratio <= 1.1:
                        score = 0.5
                    elif 0.8 <= ratio <= 1.2:
                        score = 0.25
                    else:
                        score = -1.0  # Penalize wrong answers
            except:
                # If numerical comparison fails, check for digits match
                extracted_digits = ''.join(filter(lambda x: x.isdigit(), extracted_answer))
                true_digits = ''.join(filter(lambda x: x.isdigit(), ground_truth))
                if extracted_digits == true_digits:
                    score = 0.5
                else:
                    score = -0.5  # Penalize non-matching answers
        
        scores.append(score)
    
    return scores

def reasoning_quality_reward(prompts: List[Dict], completions: List[Dict], **kwargs) -> List[float]:
    """
    Reward function for reasoning quality.
    
    Args:
        prompts: List of input prompts
        completions: List of model completions
        
    Returns:
        List of scores (higher is better)
    """
    scores = []
    
    for completion in completions:
        score = 0.0
        response = completion[0]["content"]
        
        # Extract reasoning
        extracted = extract_reasoning_and_answer(response)
        if not extracted:
            scores.append(score)
            continue
            
        reasoning = extracted["reasoning"]
        
        # Length - longer reasoning tends to be more complete
        if len(reasoning) > 500:
            score += 0.5
        elif len(reasoning) > 200:
            score += 0.25
        
        # Contains steps (numbered steps or bullet points)
        if re.search(r'(Step \d|^\d\.|\* )', reasoning, re.MULTILINE):
            score += 0.5
        
        # Contains mathematical notation
        if re.search(r'[+\-*/=]', reasoning):
            score += 0.5
        
        # Contains multiple lines of work
        if reasoning.count('\n') > 3:
            score += 0.5
        
        scores.append(score)
    
    return scores

def combined_reward(prompts: List[Dict], completions: List[Dict], answer: List[str], 
                    format_weight: float = 1.0, 
                    answer_weight: float = 2.0, 
                    reasoning_weight: float = 1.0,
                    **kwargs) -> List[float]:
    """
    Combined reward function with customizable weights.
    
    Args:
        prompts: List of input prompts
        completions: List of model completions
        answer: List of ground truth answers
        format_weight: Weight for format rewards
        answer_weight: Weight for answer rewards
        reasoning_weight: Weight for reasoning rewards
        
    Returns:
        List of weighted, combined scores
    """
    # Calculate individual rewards
    format_exact_scores = format_exact_reward(completions, **kwargs)
    format_approx_scores = format_approximate_reward(completions, **kwargs)
    answer_scores = answer_matching_reward(prompts, completions, answer, **kwargs)
    reasoning_scores = reasoning_quality_reward(prompts, completions, **kwargs)
    
    # Combine scores with weights
    combined_scores = []
    for i in range(len(completions)):
        # Use exact format score if format is correct, otherwise use approximate
        format_score = format_exact_scores[i] if format_exact_scores[i] > 0 else format_approx_scores[i]
        
        # Combined score with weights
        score = (
            format_weight * format_score + 
            answer_weight * answer_scores[i] + 
            reasoning_weight * reasoning_scores[i]
        )
        combined_scores.append(score)
        
    return combined_scores

def convert_to_jax_rewards(rewards: List[float]) -> jnp.ndarray:
    """
    Convert Python rewards to JAX array for TPU optimization.
    
    Args:
        rewards: List of reward values
        
    Returns:
        JAX array of rewards
    """
    return jnp.array(rewards, dtype=jnp.float32)

def analyze_rewards(rewards: List[float], completions: List[Dict]) -> Dict[str, Any]:
    """
    Analyze rewards for debugging and monitoring purposes.
    
    Args:
        rewards: List of reward values
        completions: List of model completions
        
    Returns:
        Dictionary with reward statistics and examples
    """
    if not rewards:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    
    # Basic statistics
    reward_array = np.array(rewards)
    stats = {
        "mean": float(np.mean(reward_array)),
        "std": float(np.std(reward_array)),
        "min": float(np.min(reward_array)),
        "max": float(np.max(reward_array)),
        "count": len(rewards)
    }
    
    # Find best and worst examples
    if len(rewards) > 0:
        best_idx = np.argmax(reward_array)
        worst_idx = np.argmin(reward_array)
        
        stats["best_example"] = {
            "reward": rewards[best_idx],
            "response": completions[best_idx][0]["content"] if best_idx < len(completions) else ""
        }
        
        stats["worst_example"] = {
            "reward": rewards[worst_idx],
            "response": completions[worst_idx][0]["content"] if worst_idx < len(completions) else ""
        }
    
    return stats