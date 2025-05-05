

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt templates for various tasks and models.

This module contains functions to create standardized prompts
for different tasks and models, ensuring consistent formatting.
"""

import logging
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)

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

def create_gemma_prompt(
    question: str, 
    instruction: Optional[str] = None, 
    include_image: bool = True
) -> str:
    """
    Format a plain text prompt with Gemma 3 tokens.
    
    Args:
        question: The user's question
        instruction: Optional instruction to include before the question
        include_image: Whether to include image tokens
        
    Returns:
        Formatted prompt for Gemma 3
    """
    tokens = GemmaTokens()
    
    # Format as a chat with user turn
    prompt = f"{tokens.START_OF_TURN}{tokens.USER}\n"
    
    # Add instruction if provided
    if instruction:
        prompt += f"{instruction}\n"
    
    # Add question
    prompt += f"{question}\n"
    
    # Add image placeholder if requested
    if include_image:
        prompt += f"{tokens.START_OF_IMAGE}\n"
    
    # Close user turn and start model turn
    prompt += f"{tokens.END_OF_TURN}\n{tokens.START_OF_TURN}{tokens.MODEL}"
    
    return prompt

def create_mathvista_prompt(
    question: str, 
    instruction: Optional[str] = "Solve this problem step-by-step",
    include_image: bool = True,
    use_format_tags: bool = True
) -> str:
    """
    Create a prompt specifically formatted for math problems with images.
    
    Args:
        question: The math problem
        instruction: Optional instruction to include
        include_image: Whether to include image tokens
        use_format_tags: Whether to include format tags in the instruction
        
    Returns:
        Formatted prompt for math problem
    """
    # Special format tags for math reasoning
    reasoning_start = "<start_reasoning>"
    reasoning_end = "<end_reasoning>"
    solution_start = "<answer>"
    solution_end = "</answer>"
    
    # Create the instruction with format tags
    if use_format_tags:
        full_instruction = (
            f"{instruction}\n\n"
            f"Look carefully at the image and solve the problem step by step.\n"
            f"First, provide your detailed reasoning between {reasoning_start} and {reasoning_end}.\n"
            f"Then, provide your final answer between {solution_start} and {solution_end}."
        )
    else:
        full_instruction = instruction
    
    # Use the base function to create the prompt
    return create_gemma_prompt(
        question=question,
        instruction=full_instruction,
        include_image=include_image
    )

def format_gemma_chat(messages: List[Dict[str, str]], include_image: bool = False) -> str:
    """
    Format a conversation as a Gemma 3 chat.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        include_image: Whether to include image token in the last user message
        
    Returns:
        Formatted Gemma 3 chat
    """
    tokens = GemmaTokens()
    formatted_chat = ""
    
    for i, message in enumerate(messages):
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role.lower() == "system":
            # System messages are prepended to the first user message
            continue
        
        # Map roles to Gemma tokens
        if role.lower() == "user":
            gemma_role = tokens.USER
        elif role.lower() in ["assistant", "model"]:
            gemma_role = tokens.MODEL
        else:
            logger.warning(f"Unknown role: {role}, using 'user' instead")
            gemma_role = tokens.USER
        
        # Add system message to the first user message
        if role.lower() == "user" and i > 0 and messages[i-1].get("role", "").lower() == "system":
            content = messages[i-1].get("content", "") + "\n\n" + content
        
        # Add turn start
        formatted_chat += f"{tokens.START_OF_TURN}{gemma_role}\n"
        
        # Add content
        formatted_chat += f"{content}\n"
        
        # Add image token if this is the last user message and include_image is True
        if role.lower() == "user" and include_image and i == len(messages) - 1:
            formatted_chat += f"{tokens.START_OF_IMAGE}\n"
        
        # Add turn end
        formatted_chat += f"{tokens.END_OF_TURN}\n"
    
    # Add start of model turn for generation
    formatted_chat += f"{tokens.START_OF_TURN}{tokens.MODEL}"
    
    return formatted_chat