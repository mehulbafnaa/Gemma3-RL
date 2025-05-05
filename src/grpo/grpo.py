#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group Relative Policy Optimization (GRPO) for mathematical reasoning with Gemma 3.

This module implements GRPO for fine-tuning Gemma 3 models to improve
mathematical reasoning capabilities, using JAX for efficient training on TPUs.
"""

import logging
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

import jax
import jax.numpy as jnp
import optax
import numpy as np
from tqdm import tqdm
from gemma import gm

from src.reward.math_reward import (
    combined_reward, 
    format_exact_reward, 
    format_approximate_reward,
    answer_matching_reward,
    reasoning_quality_reward,
    convert_to_jax_rewards,
    analyze_rewards,
    REASONING_START,
    REASONING_END,
    SOLUTION_START,
    SOLUTION_END
)

logger = logging.getLogger(__name__)

class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer for mathematical reasoning
    with Gemma 3 model.
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        output_dir: str = "outputs",
        learning_rate: float = 5e-6,
        kl_coef: float = 0.1,
        max_grad_norm: float = 0.1,
        num_generations: int = 3,
        max_seq_length: int = 1024,
        max_prompt_length: int = 256,
        format_weight: float = 1.0,
        answer_weight: float = 2.0,
        reasoning_weight: float = 1.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.99,
        weight_decay: float = 0.1,
        warmup_ratio: float = 0.1,
        logging_steps: int = 1,
        save_steps: int = 50,
        use_pjit: bool = True,
    ):
        """
        Initialize GRPO trainer
        
        Args:
            model_path: Path to Gemma model checkpoint
            tokenizer_path: Path to tokenizer model
            output_dir: Directory to save model checkpoints
            learning_rate: Learning rate for optimizer
            kl_coef: KL divergence coefficient for policy constraint
            max_grad_norm: Maximum gradient norm for clipping
            num_generations: Number of generations per prompt
            max_seq_length: Maximum sequence length
            max_prompt_length: Maximum prompt length
            format_weight: Weight for format rewards
            answer_weight: Weight for answer rewards
            reasoning_weight: Weight for reasoning rewards
            adam_beta1: Beta1 parameter for Adam optimizer
            adam_beta2: Beta2 parameter for Adam optimizer
            weight_decay: Weight decay for optimizer
            warmup_ratio: Ratio of steps for learning rate warmup
            logging_steps: Number of steps between logging
            save_steps: Number of steps between saving checkpoints
            use_pjit: Whether to use pjit for distributed training
        """
        # Import here to avoid circular imports
        from gemma import gm
        import jax
        
        # Setup paths
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.kl_coef = kl_coef
        self.max_grad_norm = max_grad_norm
        self.num_generations = num_generations
        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length
        self.format_weight = format_weight
        self.answer_weight = answer_weight
        self.reasoning_weight = reasoning_weight
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.use_pjit = use_pjit
        
        # Initialize model components
        logger.info("Loading model components...")
        self.tokenizer = gm.text.Gemma3Tokenizer(path=self.tokenizer_path)
        self.model = gm.nn.Gemma3_4B()
        
        # Load parameters
        logger.info(f"Loading parameters from {self.model_path}...")
        self.params = gm.ckpts.load_params(path=self.model_path, text_only=False)
        
        # Convert parameters to bfloat16 for TPU compatibility
        self.params = self._ensure_bfloat16(self.params)
        
        # Create reference parameters (frozen copy)
        self.ref_params = jax.tree_util.tree_map(lambda x: x, self.params)
        
        # Initialize optimizer
        self.optimizer, self.opt_state = self._create_optimizer()
        
        # Initialize JAX random key
        self.rng = jax.random.PRNGKey(42)
        
        # Initialize training state
        self.global_step = 0
        
        logger.info("GRPO Trainer initialized")
    
    def _ensure_bfloat16(self, params):
        """Convert floating point arrays to bfloat16"""
        if isinstance(params, dict):
            return {k: self._ensure_bfloat16(v) for k, v in params.items()}
        elif isinstance(params, (list, tuple)):
            return type(params)(self._ensure_bfloat16(v) for v in params)
        elif hasattr(params, 'dtype') and jnp.issubdtype(params.dtype, jnp.floating):
            if params.dtype != jnp.bfloat16:
                return jnp.asarray(params, dtype=jnp.bfloat16)
        return params
    
    def _create_optimizer(self):
        """Create optimizer with learning rate schedule"""
        # Create learning rate schedule
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.learning_rate,
            warmup_steps=int(1000 * self.warmup_ratio), # Assuming 1000 total steps
            decay_steps=1000,
            end_value=0.1 * self.learning_rate
        )
        
        # Create optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adamw(
                learning_rate=lr_schedule,
                b1=self.adam_beta1,
                b2=self.adam_beta2,
                weight_decay=self.weight_decay
            )
        )
        
        # Initialize optimizer state
        opt_state = optimizer.init(self.params)
        
        return optimizer, opt_state
    
    def format_prompt(self, question: str, include_system_prompt: bool = True):
        """
        Format a question with Gemma tokens and special reasoning tokens
        
        Args:
            question: The math problem question
            include_system_prompt: Whether to include system prompt
            
        Returns:
            Formatted prompt
        """
        # Special tokens for Gemma 3
        START_OF_TURN = "<start_of_turn>"
        END_OF_TURN = "<end_of_turn>"
        START_OF_IMAGE = "<start_of_image>"
        USER = "user"
        MODEL = "model"
        
        # Define system prompt with our special formatting tokens
        system_instruction = (
            f"Solve this mathematical problem step-by-step and give me all the reasoning traces. "
            f"Place your reasoning between {REASONING_START} and {REASONING_END}. "
            f"Then, provide your final answer between {SOLUTION_START} and {SOLUTION_END}."
        )
        
        # Create formatted prompt with Gemma's special tokens
        prompt = f"{START_OF_TURN}{USER}\n"
        
        if include_system_prompt:
            prompt += f"{system_instruction}\n"
        
        prompt += f"{question}\n"
        prompt += f"{START_OF_IMAGE}\n"  # Always include image token for MathVista
        prompt += f"{END_OF_TURN}\n{START_OF_TURN}{MODEL}"
        
        return prompt
    
    def get_logits(self, params, tokens, images=None):
        """
        Get logits from model for tokens and images
        
        Args:
            params: Model parameters
            tokens: Input token IDs
            images: Optional image inputs
            
        Returns:
            Model logits
        """
        # Forward pass with model to get logits
        model_output = self.model.apply(
            {'params': params}, 
            tokens=tokens, 
            images=images,
            return_last_only=False,  # Get all token logits
        )
        return model_output.logits
    
    def compute_log_probs(self, logits, tokens):
        """
        Compute log probabilities for tokens given logits
        
        Args:
            logits: Model logits from forward pass
            tokens: Token IDs
            
        Returns:
            Log probabilities
        """
        # Shift logits and tokens for next-token prediction
        logits = logits[:, :-1, :]  # Remove last position
        next_tokens = tokens[:, 1:]  # Targets are next tokens
        
        # Convert logits to log probabilities
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        
        # Get log probs for actual next tokens
        batch_size, seq_len = next_tokens.shape
        indices = jnp.arange(batch_size)[:, None]
        indices = jnp.repeat(indices, seq_len, axis=1)
        token_indices = jnp.stack([indices, jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0), next_tokens], axis=-1)
        token_log_probs = jnp.take_along_axis(
            log_probs, token_indices[..., 2][..., None], axis=-1
        ).squeeze(-1)
        
        return token_log_probs
    
    def compute_kl_divergence(self, logits_pi, logits_ref):
        """
        Compute KL divergence between policy and reference model
        
        Args:
            logits_pi: Logits from policy model
            logits_ref: Logits from reference model
            
        Returns:
            KL divergence
        """
        # Convert to probabilities
        probs_pi = jax.nn.softmax(logits_pi, axis=-1)
        log_probs_pi = jax.nn.log_softmax(logits_pi, axis=-1)
        log_probs_ref = jax.nn.log_softmax(logits_ref, axis=-1)
        
        # Compute KL: sum_i p_i * (log p_i - log q_i)
        kl = jnp.sum(probs_pi * (log_probs_pi - log_probs_ref), axis=-1)
        return kl
    
    def generate_response(self, params, prompt, image=None):
        """
        Generate a response for a prompt
        
        Args:
            params: Model parameters
            prompt: Text prompt
            image: Optional image input
            
        Returns:
            Generated response and logits
        """
        
        
        # Create new RNG key
        self.rng, subkey = jax.random.split(self.rng)
        
        # Create sampler for generation
        sampler = gm.text.ChatSampler(
            model=self.model,
            params=params,
            tokenizer=self.tokenizer,
            multi_turn=False
        )
        
        # Generate response using the ChatSampler API
        response_text = sampler.chat(
            prompt=prompt,
            images=image,
            rng=subkey
        )
        
        # Get response tokens (for getting logits)
        prompt_tokens = jnp.array(self.tokenizer.encode(prompt, add_bos=True))[None, :]
        response_tokens = jnp.array(self.tokenizer.encode(response_text, add_bos=False))[None, :]
        full_tokens = jnp.concatenate([prompt_tokens, response_tokens], axis=1)
        
        # Get logits for the tokens
        logits = self.get_logits(params, full_tokens, image)
        
        return response_text, logits, full_tokens
    
    def generate_responses(self, params, prompts, images=None):
        """
        Generate multiple responses for prompts
        
        Args:
            params: Model parameters
            prompts: List of text prompts
            images: Optional list of image inputs
            
        Returns:
            List of generated responses and their logits
        """
        responses = []
        response_logits = []
        response_tokens = []
        
        for i, prompt in enumerate(prompts):
            prompt_image = None if images is None else images[i]
            
            # Generate multiple responses for this prompt
            prompt_responses = []
            prompt_logits = []
            prompt_tokens = []
            
            for _ in range(self.num_generations):
                # Generate response
                response, logits, tokens = self.generate_response(
                    params, prompt, prompt_image
                )
                
                prompt_responses.append(response)
                prompt_logits.append(logits)
                prompt_tokens.append(tokens)
            
            responses.append(prompt_responses)
            response_logits.append(prompt_logits)
            response_tokens.append(prompt_tokens)
        
        return responses, response_logits, response_tokens
    
    def prepare_completions(self, responses):
        """
        Prepare completions in the format expected by reward functions
        
        Args:
            responses: List of generated responses
            
        Returns:
            List of completions
        """
        completions = []
        
        for batch_responses in responses:
            batch_completions = []
            for response in batch_responses:
                batch_completions.append([{"role": "assistant", "content": response}])
            completions.append(batch_completions)
        
        return completions
    
    def prepare_prompt_dicts(self, prompts):
        """
        Prepare prompts in the format expected by reward functions
        
        Args:
            prompts: List of text prompts
            
        Returns:
            List of prompt dicts
        """
        prompt_dicts = []
        
        for prompt in prompts:
            prompt_dicts.append([{"role": "user", "content": prompt}])
        
        return prompt_dicts
    
    def compute_rewards(self, prompts, responses, answers):
        """
        Compute rewards for generated responses
        
        Args:
            prompts: Original prompts
            responses: Generated responses
            answers: Ground truth answers
            
        Returns:
            Rewards for each response
        """
        # Prepare data for reward functions
        prompt_dicts = self.prepare_prompt_dicts(prompts)
        completions = self.prepare_completions(responses)
        
        batch_rewards = []
        
        # Process each batch
        for i in range(len(prompts)):
            # Prepare inputs for this batch
            batch_completions = completions[i]
            batch_prompt = prompt_dicts[i]
            batch_answer = [answers[i]] * len(batch_completions)
            
            # Compute rewards using combined reward function
            rewards = combined_reward(
                prompts=[batch_prompt] * len(batch_completions),
                completions=batch_completions,
                answer=batch_answer,
                format_weight=self.format_weight,
                answer_weight=self.answer_weight,
                reasoning_weight=self.reasoning_weight
            )
            
            batch_rewards.append(rewards)
        
        return batch_rewards
    
    def train_step(self, prompts, answers, images=None):
        """
        Perform one GRPO training step
        
        Args:
            prompts: Text prompts
            answers: Ground truth answers
            images: Optional image inputs
            
        Returns:
            Updated parameters and metrics
        """
        # Save reference to old parameters
        ref_params = self.params
        
        # Generate responses with current policy
        responses, response_logits, response_tokens = self.generate_responses(
            self.params, prompts, images
        )
        
        # Compute rewards
        batch_rewards = self.compute_rewards(prompts, responses, answers)
        
        # Prepare for gradient computation
        flat_rewards = []
        flat_logits = []
        flat_tokens = []
        flat_ref_logits = []
        
        # Flatten batch structure
        for batch_idx in range(len(prompts)):
            for sample_idx in range(self.num_generations):
                flat_rewards.append(batch_rewards[batch_idx][sample_idx])
                flat_logits.append(response_logits[batch_idx][sample_idx])
                flat_tokens.append(response_tokens[batch_idx][sample_idx])
                
                # Compute reference logits
                ref_logits = self.get_logits(
                    ref_params, 
                    response_tokens[batch_idx][sample_idx],
                    None if images is None else images[batch_idx]
                )
                flat_ref_logits.append(ref_logits)
        
        # Convert to JAX arrays
        flat_rewards = convert_to_jax_rewards(flat_rewards)
        
        # Define loss function
        def loss_fn(params):
            """Compute GRPO loss for policy optimization"""
            policy_logits = []
            kl_divs = []
            
            # Recompute policy logits
            for i, tokens in enumerate(flat_tokens):
                policy_output = self.get_logits(params, tokens, None if images is None else images[i // self.num_generations])
                policy_logits.append(policy_output)
                
                # Compute KL divergence
                kl = self.compute_kl_divergence(policy_output, flat_ref_logits[i])
                kl_divs.append(jnp.mean(kl))
            
            # Mean KL divergence
            mean_kl = jnp.mean(jnp.array(kl_divs))
            
            # Compute policy gradient loss (negative because we want to maximize reward)
            pg_loss = -jnp.mean(flat_rewards)
            
            # Total loss with KL penalty
            total_loss = pg_loss + self.kl_coef * mean_kl
            
            return total_loss, (pg_loss, mean_kl, jnp.mean(flat_rewards))
        
        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (total_loss, (pg_loss, kl_div, mean_reward)), grads = grad_fn(self.params)
        
        # Update parameters
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, self.params
        )
        self.params = optax.apply_updates(self.params, updates)
        
        # Create metrics dictionary
        metrics = {
            "loss": float(total_loss),
            "pg_loss": float(pg_loss),
            "kl_div": float(kl_div),
            "mean_reward": float(mean_reward),
            "completion_length": float(np.mean([len(r) for batch_r in responses for r in batch_r]))
        }
        
        # Add reward analysis
        flat_completions = []
        for batch_completions in self.prepare_completions(responses):
            flat_completions.extend(batch_completions)
        reward_analysis = analyze_rewards(list(flat_rewards), flat_completions)
        metrics.update({f"reward_{k}": v for k, v in reward_analysis.items() if k in ["mean", "std", "min", "max"]})
        
        return self.params, metrics
    
    def save_checkpoint(self, step):
        """Save model checkpoint"""

        
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint to {checkpoint_dir}")
        gm.ckpts.save_params(self.params, path=str(checkpoint_dir))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        
        # Save config with training arguments
        import json
        with open(checkpoint_dir / "grpo_config.json", "w") as f:
            config = {
                "learning_rate": self.learning_rate,
                "kl_coef": self.kl_coef,
                "max_grad_norm": self.max_grad_norm,
                "num_generations": self.num_generations,
                "max_seq_length": self.max_seq_length,
                "max_prompt_length": self.max_prompt_length,
                "format_weight": self.format_weight,
                "answer_weight": self.answer_weight,
                "reasoning_weight": self.reasoning_weight,
                "step": step
            }
            json.dump(config, f, indent=2)
    
    def train(self, dataset, num_epochs=1, max_steps=None, batch_size=1):
        """
        Train model using GRPO
        
        Args:
            dataset: Training dataset with prompts and answers
            num_epochs: Number of training epochs
            max_steps: Maximum number of steps to train for (overrides num_epochs)
            batch_size: Batch size
            
        Returns:
            Trained model parameters
        """
        logger.info(f"Starting GRPO training for {num_epochs} epochs")
        
        # Track training start time
        start_time = time.time()
        
        # Total steps counter
        total_steps = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
            
            # Process dataset in batches
            for batch_idx in range(0, len(dataset), batch_size):
                if max_steps is not None and total_steps >= max_steps:
                    logger.info(f"Reached maximum steps ({max_steps}), stopping training")
                    break
                
                # Get batch data
                batch_data = dataset[batch_idx:batch_idx + batch_size]
                
                # Extract prompts and answers from batch
                prompts = []
                answers = []
                images = []
                
                for item in batch_data:
                    # Format prompts with special tokens
                    prompts.append(self.format_prompt(item["question"]))
                    answers.append(item["answer"])
                    
                    # Extract image if available
                    if "image" in item:
                        images.append(item["image"])
                
                if not images:
                    images = None
                
                # Perform training step
                _, metrics = self.train_step(prompts, answers, images)
                
                # Update step counters
                self.global_step += 1
                total_steps += 1
                
                # Log training progress
                if self.global_step % self.logging_steps == 0:
                    # Calculate elapsed time
                    elapsed = time.time() - start_time
                    steps_per_second = self.global_step / elapsed
                    
                    # Log metrics
                    log_str = f"Step: {self.global_step} | "
                    log_str += f"Loss: {metrics['loss']:.6f} | "
                    log_str += f"Reward: {metrics['mean_reward']:.4f} (±{metrics['reward_std']:.4f}) | "
                    log_str += f"KL: {metrics['kl_div']:.6f} | "
                    log_str += f"Steps/sec: {steps_per_second:.2f}"
                    
                    logger.info(log_str)
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint(self.global_step)
        
        # Save final checkpoint
        self.save_checkpoint(self.global_step)
        
        # Log training summary
        total_time = time.time() - start_time
        logger.info(f"Training completed. Total steps: {self.global_step}, Total time: {total_time:.2f}s")
        
        return self.params
    
    def generate_math_example(self, question, answer=None, image=None):
        """
        Generate math reasoning example from trained model
        
        Args:
            question: Math problem question
            answer: Optional ground truth answer for evaluation
            image: Optional image for multimodal problems
            
        Returns:
            Generated response and evaluation metrics
        """
        # Format prompt
        prompt = self.format_prompt(question)
        
        # Generate response
        response, _, _ = self.generate_response(self.params, prompt, image)
        
        # Evaluate if answer is provided
        metrics = {}
        if answer is not None:
            # Prepare inputs for reward function
            prompt_dict = [{"role": "user", "content": question}]
            completion = [{"role": "assistant", "content": response}]
            
            # Compute format reward
            format_score = format_exact_reward([completion])[0]
            metrics["format_score"] = format_score
            
            # Compute answer correctness
            answer_score = answer_matching_reward([prompt_dict], [completion], [answer])[0]
            metrics["answer_score"] = answer_score
            
            # Compute reasoning quality
            reasoning_score = reasoning_quality_reward([prompt_dict], [completion])[0]
            metrics["reasoning_score"] = reasoning_score
            
            # Combine scores
            metrics["total_score"] = (
                self.format_weight * format_score +
                self.answer_weight * answer_score +
                self.reasoning_weight * reasoning_score
            )
        
        return response, metrics