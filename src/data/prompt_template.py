# -*- coding: utf-8 -*-
"""
prompt_templates.py - Multimodal prompt templates for Gemma 3 and MathVista

This module provides optimized prompt templates for multimodal mathematical
reasoning with Gemma 3, specifically tailored for the MathVista dataset.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Special token strings for Gemma 3
class GemmaTokens:
    """String representations of Gemma 3 special tokens"""
    BOS = "<bos>"
    EOS = "<eos>"
    START_OF_TURN = "<start_of_turn>"
    END_OF_TURN = "<end_of_turn>"
    START_OF_IMAGE = "<start_of_image>"
    END_OF_IMAGE = "<end_of_image>"
    USER = "user"
    MODEL = "model"

def create_mathvista_prompt(
    question: str,
    instruction: str = "Solve this problem step-by-step and give me all the reasoning traces.",
    include_image: bool = True,
    include_bos: bool = False  # Set to False as chat_sampler likely adds this
) -> str:
    """
    Create a formatted prompt for Gemma 3 with a MathVista problem.
    
    Args:
        question: The MathVista question text
        instruction: Instruction for solving the problem
        include_image: Whether to include the image token
        include_bos: Whether to include the BOS token
        
    Returns:
        Formatted prompt string following the successful Gemma 3 pattern
    """
    tokens = GemmaTokens()
    
    # Combine question and instruction
    if question and instruction and instruction not in question:
        full_query = f"{question.strip()}\n\n{instruction}"
    else:
        full_query = question.strip()
    
    # Format for Gemma 3
    prompt = f"{tokens.START_OF_TURN}{tokens.USER}\n"
    prompt += f"{full_query}\n"
    
    # Add image token if needed
    if include_image:
        prompt += f"{tokens.START_OF_IMAGE}\n"
    
    prompt += f"{tokens.END_OF_TURN}\n{tokens.START_OF_TURN}{tokens.MODEL}"
    
    # Add BOS token if requested
    if include_bos:
        prompt = f"{tokens.BOS}{prompt}"
    
    return prompt

def extract_answer_from_response(response: str, answer_tag: str = "answer") -> Optional[str]:
    """
    Extract the answer from a model's response using the specified tag.
    
    Args:
        response: The model's response text
        answer_tag: The tag used to mark the answer
        
    Returns:
        The extracted answer or None if not found
    """
    import re
    
    # Pattern to find content between tags
    pattern = rf'<{answer_tag}>(.*?)</{answer_tag}>'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    # Fallback pattern for final answer without tags
    fallback_patterns = [
        r'final answer:?\s*(.*?)(?:$|\n)',
        r'answer:?\s*(.*?)(?:$|\n)',
        r'therefore,?\s*(.*?)(?:$|\n)'
    ]
    
    for pattern in fallback_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None