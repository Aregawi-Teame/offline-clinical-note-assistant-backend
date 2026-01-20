"""
Service for running the language model with Hugging Face Transformers.
"""
from typing import Optional
import torch

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# Global singleton instances for model and tokenizer
_global_model = None
_global_tokenizer = None
_global_device = None
_global_model_id = None


def _resolve_device(device_setting: str) -> str:
    """
    Resolve device setting to actual device string.
    
    Note: MPS (Apple Silicon GPU) is automatically avoided due to known issues
    with model.generate() hanging. Use DEVICE=cpu explicitly to suppress warnings.
    
    Args:
        device_setting: Device setting from config (auto/cuda/mps/cpu)
        
    Returns:
        Resolved device string (cuda/cpu, never mps due to generation issues)
    """
    if device_setting == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS has known issues with model.generate() hanging
            # Automatically fall back to CPU for stability
            logger.warning(
                "MPS (Apple Silicon GPU) detected but auto-disabled due to known "
                "generation issues. Using CPU instead. Set DEVICE=cpu explicitly "
                "to suppress this warning."
            )
            return "cpu"
        else:
            return "cpu"
    elif device_setting == "mps":
        # User explicitly requested MPS, but we know it has issues
        logger.warning(
            "MPS device explicitly requested, but MPS is known to hang during "
            "model.generate(). Falling back to CPU for stability. "
            "To suppress this warning, use DEVICE=cpu."
        )
        return "cpu"
    return device_setting


def _load_model_and_tokenizer(model_id: str, device: str) -> tuple[torch.nn.Module, object]:
    """
    Load model and tokenizer lazily (singleton-style).
    
    Args:
        model_id: HuggingFace model identifier
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        RuntimeError: If model loading fails (e.g., OOM)
        ImportError: If transformers library is not installed
    """
    global _global_model, _global_tokenizer, _global_device, _global_model_id
    
    # Return cached instances if already loaded for this model_id and device
    if _global_model is not None and _global_tokenizer is not None:
        if _global_model_id == model_id and _global_device == device:
            logger.debug(f"Using cached model and tokenizer for {model_id} on {device}")
            return _global_model, _global_tokenizer
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers library is not installed. "
            "Install it with: pip install transformers torch"
        )
    
    logger.info(f"Loading model: {model_id} on device: {device}")
    
    try:
        # Get HuggingFace token from environment if available
        import os
        hf_token = os.environ.get('HF_TOKEN')
        
        # Prepare model loading kwargs
        # Use device_map="auto" for CUDA to let HuggingFace handle device placement
        # For CPU, use .to(device) after loading (simpler and more reliable)
        use_device_map = device == "cuda"
        
        model_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
        }
        
        # Add token if available (for private or gated models)
        if hf_token:
            model_kwargs["token"] = hf_token
            logger.debug("Using HuggingFace token for authentication")
        
        # Add device_map only for CUDA (let HuggingFace handle device placement)
        if use_device_map:
            model_kwargs["device_map"] = "auto"
            logger.debug("Using device_map='auto' for CUDA device placement")
        
        # Load tokenizer
        logger.debug("Loading tokenizer...")
        if hf_token:
            _global_tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        else:
            _global_tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Ensure pad_token is set if tokenizer doesn't have one (use eos_token as fallback)
        # This is important for proper tokenization, but we won't use pad_token in generation
        if _global_tokenizer.pad_token is None:
            if _global_tokenizer.eos_token is not None:
                _global_tokenizer.pad_token = _global_tokenizer.eos_token
                logger.debug(f"Set pad_token to eos_token: {_global_tokenizer.eos_token}")
            else:
                logger.warning("Tokenizer has no pad_token or eos_token - this may cause issues")
        
        # Load model
        logger.debug("Loading model (this may take a while)...")
        _global_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs
        )
        
        # Only move to device manually if NOT using device_map (i.e., for CPU)
        # Using device_map="auto" and .to(device) together causes RuntimeError
        if not use_device_map:
            _global_model = _global_model.to(device)
            logger.debug(f"Manually moved model to {device}")
        else:
            logger.debug("Model device placement handled by device_map='auto'")
        
        # Set model to evaluation mode
        _global_model.eval()
        
        # Cache the model_id and device
        _global_model_id = model_id
        _global_device = device
        
        logger.info(f"Model {model_id} loaded successfully on {device}")
        
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "oom" in error_msg:
            raise RuntimeError(
                f"Out of memory error while loading model {model_id}. "
                f"Try: (1) Use a smaller model, (2) Use CPU device, "
                f"(3) Reduce MAX_NEW_TOKENS, or (4) Free up GPU memory."
            ) from e
        else:
            raise RuntimeError(
                f"Failed to load model {model_id}: {e}. "
                f"Check that the model ID is correct and you have sufficient resources."
            ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error loading model {model_id}: {e}"
        ) from e
    
    return _global_model, _global_tokenizer


class ModelRunner:
    """Run the language model to generate clinical notes."""
    
    def __init__(self):
        """Initialize model runner with configuration."""
        self.model_id = settings.MODEL_ID
        self.device_setting = settings.DEVICE
        self.default_max_new_tokens = settings.MAX_NEW_TOKENS
        self.default_temperature = settings.TEMPERATURE
        self.default_top_p = settings.TOP_P
        
        # Resolve device
        self.device = _resolve_device(self.device_setting)
        logger.debug(f"Device resolved: {self.device_setting} -> {self.device}")
    
    def _ensure_model_loaded(self):
        """Ensure model and tokenizer are loaded (lazy loading)."""
        if _global_model is None or _global_tokenizer is None:
            _load_model_and_tokenizer(self.model_id, self.device)
    
    def get_model_id(self) -> str:
        """
        Get the model identifier.
        
        Returns:
            Model ID string
        """
        return self.model_id
    
    def is_model_loaded(self) -> bool:
        """
        Check if model is currently loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        if settings.DEMO_MODE:
            return False  # Demo mode doesn't load models
        return _global_model is not None
    
    def _detect_task_from_prompt(self, prompt: str) -> str:
        """
        Detect task type from prompt content.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Task type string (SOAP, DISCHARGE, or REFERRAL)
        """
        prompt_upper = prompt.upper()
        if "SOAP" in prompt_upper or "SOAP NOTE" in prompt_upper:
            return "SOAP"
        elif "DISCHARGE" in prompt_upper or "DISCHARGE SUMMARY" in prompt_upper:
            return "DISCHARGE"
        elif "REFERRAL" in prompt_upper or "REFERRAL LETTER" in prompt_upper:
            return "REFERRAL"
        else:
            return "SOAP"  # Default to SOAP
    
    def _get_demo_response(self, task: str) -> str:
        """
        Get deterministic stubbed response for demo mode.
        
        Args:
            task: Task type (SOAP, DISCHARGE, or REFERRAL)
            
        Returns:
            Stubbed response string with all required sections
        """
        if task == "SOAP":
            return """SUBJECTIVE:
Chief Complaint: Not provided
History of Present Illness: Not provided
Review of Systems: Not provided
Past Medical History: Not provided
Medications: Not provided
Allergies: Not provided
Social History: Not provided

OBJECTIVE:
Vital Signs: Not provided
Physical Examination:
  - General Appearance: Not provided

ASSESSMENT:
Primary Diagnosis: Not provided
Differential Diagnoses: Not provided

PLAN:
Treatment Plan: Not provided
Medications: Not provided
Follow-up: Not provided
Patient Education: Not provided

Clarifying questions (max 3):
None required."""
        elif task == "DISCHARGE":
            return """ADMISSION INFORMATION:
Admission Date: Not provided
Admitting Diagnosis: Not provided
Admitting Service: Not provided

HOSPITAL COURSE:
Hospital Course: Not provided
Procedures Performed: Not provided
Consultations: Not provided
Complications: Not provided

DISCHARGE INFORMATION:
Discharge Date: Not provided
Discharge Diagnosis: Not provided
Discharge Condition: Not provided

DISCHARGE MEDICATIONS:
Not provided

DISCHARGE INSTRUCTIONS:
Activity: Not provided
Diet: Not provided
Wound Care: Not provided
Other Instructions: Not provided

FOLLOW-UP:
Primary Care Provider Follow-up: Not provided
Specialty Follow-up: Not provided
Labs/Imaging Pending: Not provided

Clarifying questions (max 3):
None required."""
        else:  # REFERRAL
            return """To: Not specified
From: Not provided
Date: Not provided
Re: Patient descriptor - minimize PHI

REASON FOR REFERRAL:
Not provided

RELEVANT HISTORY:
Current Condition: Not provided
Past Medical History: Not provided
Surgical History: Not provided
Family History: Not provided

CURRENT MEDICATIONS:
Not provided

CURRENT TREATMENTS:
Not provided

RELEVANT DIAGNOSTICS:
Laboratory Results: Not provided
Imaging Results: Not provided
Other Studies: Not provided

SPECIFIC QUESTIONS OR CONCERNS:
Not provided

URGENCY:
Urgency Level: Not provided
Timeline: Not provided

Clarifying questions (max 3):
None required."""
    
    def run(self, prompt: str, options: Optional[dict] = None) -> str:
        """
        Run the model with the given prompt and options.
        
        Args:
            prompt: Input prompt string
            options: Optional dict with generation parameters:
                - max_new_tokens: int (defaults to config value)
                - temperature: float (defaults to config value)
                - top_p: float (defaults to config value)
        
        Returns:
            Generated text only (prompt stripped if present)
            
        Raises:
            RuntimeError: If generation fails (e.g., OOM)
        """
        # Check if demo mode is enabled
        if settings.DEMO_MODE:
            task = self._detect_task_from_prompt(prompt)
            logger.info(f"DEMO_MODE enabled: Returning stubbed response for task: {task}")
            return self._get_demo_response(task)
        
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        # Get options or use defaults
        if options is None:
            options = {}
        
        max_new_tokens = options.get("maxTokens", self.default_max_new_tokens)
        temperature = options.get("temperature", self.default_temperature)
        top_p = options.get("topP", self.default_top_p)
        
        logger.info(
            f"Starting inference: prompt_len={len(prompt)}, "
            f"max_new_tokens={max_new_tokens}, "
            f"temperature={temperature}, top_p={top_p}, "
            f"device={self.device}"
        )
        
        try:
            # Tokenize input
            logger.debug("Step 1: Tokenizing input prompt...")
            # Tokenize and move to device
            # Note: attention_mask is automatically included in inputs dict
            inputs = _global_tokenizer(prompt, return_tensors="pt").to(self.device)
            input_length = inputs["input_ids"].shape[1]
            
            # Log tokenization details for debugging BOS/EOS token issues
            # Check if BOS token was added (some tokenizers add it, some don't)
            if hasattr(_global_tokenizer, 'bos_token_id') and _global_tokenizer.bos_token_id is not None:
                first_token_id = inputs["input_ids"][0, 0].item()
                if first_token_id == _global_tokenizer.bos_token_id:
                    logger.debug(f"BOS token detected at start of input (token_id: {first_token_id})")
            
            logger.debug(f"Tokenization complete. Input shape: {inputs['input_ids'].shape}, input_length={input_length}")
            logger.debug(f"Attention mask shape: {inputs.get('attention_mask', 'not provided').shape if hasattr(inputs.get('attention_mask', None), 'shape') else 'not provided'}")
            
            # Prepare generation kwargs
            # For low temperatures (<= 0.2), use greedy decoding to avoid numerical issues
            # This is especially important for MPS and some GPU setups
            # Temperature 0.2 is default, so we use greedy by default (more stable)
            use_sampling = temperature > 0.2
            
            # Get token IDs
            eos_token_id = _global_tokenizer.eos_token_id
            pad_token_id = _global_tokenizer.pad_token_id
            
            # IMPORTANT: Do NOT pass pad_token_id to generation kwargs
            # pad_token_id is only used for input padding, not for generation
            # If passed, the model may generate pad tokens instead of stopping at EOS
            # Only set eos_token_id to ensure proper stopping
            
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "eos_token_id": eos_token_id,  # Explicitly set EOS token to stop generation properly
                "repetition_penalty": 1.1,  # Slight penalty to reduce repetition loops
            }
            
            # Only add pad_token_id if model doesn't have one set and we need it for input padding
            # But do NOT use it as a generation stopping criterion
            if pad_token_id is not None and eos_token_id is None:
                # Rare case: model has pad but no EOS - use pad for stopping (not ideal)
                logger.warning(f"Model has pad_token_id ({pad_token_id}) but no eos_token_id - using pad for stopping")
                generation_kwargs["eos_token_id"] = pad_token_id
            
            logger.debug(f"Generation token IDs: pad_token_id={pad_token_id} (NOT used in generation), eos_token_id={eos_token_id}")
            
            if use_sampling:
                # Ensure temperature is within safe range for sampling
                safe_temperature = max(0.3, min(temperature, 2.0))
                generation_kwargs["temperature"] = safe_temperature
                generation_kwargs["top_p"] = max(0.1, min(top_p, 1.0))
                generation_kwargs["do_sample"] = True
                logger.debug(f"Using sampling mode: temperature={safe_temperature}, top_p={generation_kwargs['top_p']}, repetition_penalty={generation_kwargs['repetition_penalty']}")
            else:
                # Greedy decoding for low/zero temperature (more stable, deterministic)
                # Don't pass temperature or top_p when using greedy decoding
                # But we still apply repetition_penalty to avoid loops
                generation_kwargs["do_sample"] = False
                logger.debug(f"Using greedy decoding mode (temperature <= 0.2), repetition_penalty={generation_kwargs['repetition_penalty']}")
            
            logger.info(f"Step 2: Starting model.generate() on device {self.device} with kwargs: {generation_kwargs}")
            logger.debug(f"Model device: {next(_global_model.parameters()).device}")
            logger.info(f"Input tensor device: {inputs['input_ids'].device}")
            logger.info(f"⏳ Generation may take 2-5 minutes for {max_new_tokens} tokens on T4 GPU. Please wait...")
            
            # Device should already be resolved (MPS -> CPU) in _resolve_device
            # No need for fallback logic - device resolution handles MPS -> CPU conversion
            
            # Generate with torch.no_grad() for efficiency
            import time
            generation_start = time.time()
            with torch.no_grad():
                logger.info(f"Step 3: Calling model.generate() on {self.device} - this may take a while...")
                try:
                    outputs = _global_model.generate(
                        **inputs,
                        **generation_kwargs
                    )
                    generation_time = time.time() - generation_start
                    logger.info(f"Generation complete in {generation_time:.1f}s. Output shape: {outputs.shape}")
                except Exception as gen_error:
                    generation_time = time.time() - generation_start
                    logger.error(f"Generation failed after {generation_time:.1f}s: {gen_error}")
                    raise
            
            logger.debug("Step 4: Decoding output tokens...")
            # Decode the full output once (cache for fallback methods)
            full_text = _global_tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug(f"Decoding complete. Full text length: {len(full_text)} chars")
            logger.debug(f"Prompt length: {len(prompt)} chars, Input tokens: {input_length}")
            
            # Strip the prompt to return only generated text
            logger.info("Step 5: Stripping prompt from generated text...")
            
            # PRIMARY METHOD: Decode only the new tokens (most reliable)
            # This avoids issues with tokenizer decoding differences
            # Note: We use input_length directly - the model.generate() output includes
            # the input tokens + generated tokens, so slicing at input_length gives us only generated tokens
            # If BOS token was added by tokenizer, it's already in input_length count
            generated_text = ""
            if outputs.shape[1] > input_length:
                logger.debug(f"Method 1 (token slicing): Extracting new tokens (output: {outputs.shape[1]}, input: {input_length})")
                
                # Extract only new tokens (everything after input_length)
                # Note: input_length already accounts for BOS if tokenizer added it
                generated_tokens = outputs[0][input_length:].clone()
                num_new_tokens = len(generated_tokens)
                logger.debug(f"New tokens shape: {generated_tokens.shape}, num_tokens: {num_new_tokens}")
                
                # Check if first generated token is BOS (unlikely but possible)
                # This would indicate the model generated BOS as first token (shouldn't happen)
                if hasattr(_global_tokenizer, 'bos_token_id') and _global_tokenizer.bos_token_id is not None:
                    first_gen_token = generated_tokens[0].item() if len(generated_tokens) > 0 else None
                    if first_gen_token == _global_tokenizer.bos_token_id:
                        logger.warning(f"First generated token is BOS (token_id: {first_gen_token}) - skipping it")
                        generated_tokens = generated_tokens[1:]  # Skip BOS if model generated it
                        num_new_tokens = len(generated_tokens)
                
                # Get special token IDs to filter out
                pad_token_id = _global_tokenizer.pad_token_id
                eos_token_id = _global_tokenizer.eos_token_id
                bos_token_id = _global_tokenizer.bos_token_id
                unk_token_id = _global_tokenizer.unk_token_id
                
                # Collect all special token IDs (excluding None)
                special_token_ids = set()
                for token_id in [pad_token_id, eos_token_id, bos_token_id, unk_token_id]:
                    if token_id is not None:
                        special_token_ids.add(token_id)
                
                # Log first few token IDs for debugging
                sample_ids = generated_tokens[:min(10, num_new_tokens)].tolist()
                logger.debug(f"First 10 generated token IDs: {sample_ids}")
                
                # Filter out special tokens before decoding
                # Convert to list, filter, then back to tensor
                token_list = generated_tokens.tolist()
                filtered_tokens = [tok_id for tok_id in token_list if tok_id not in special_token_ids]
                
                # Check if ALL tokens are pad tokens (critical issue)
                unique_tokens = set(token_list)
                if len(unique_tokens) == 1 and pad_token_id in unique_tokens:
                    logger.error(f"⚠️  CRITICAL: All {num_new_tokens} generated tokens are pad_token_id ({pad_token_id})!")
                    logger.error(f"   This indicates the model generated no content - only padding.")
                    logger.error(f"   Possible causes:")
                    logger.error(f"     1. Model is not generating properly (check model config)")
                    logger.error(f"     2. Prompt may be too long or malformed")
                    logger.error(f"     3. Generation kwargs may be incorrect")
                    logger.error(f"     4. Model may need different eos_token_id or stopping criteria")
                    # Don't decode - will trigger empty text handling below
                    generated_text = ""
                elif len(filtered_tokens) == 0:
                    logger.error(f"⚠️  All {num_new_tokens} generated tokens are special tokens (pad/eos/bos/unk). Model generated no meaningful content!")
                    logger.error(f"   Unique token IDs generated: {unique_tokens}")
                    if pad_token_id and pad_token_id in unique_tokens:
                        pad_count = token_list.count(pad_token_id)
                        logger.error(f"   Model generated pad_token_id ({pad_token_id}) {pad_count}/{num_new_tokens} times - this should not happen!")
                        logger.error(f"   Possible causes: pad_token_id was passed to generation kwargs (should only use eos_token_id)")
                        logger.error(f"   Or model has generation config issues")
                    if eos_token_id and eos_token_id in unique_tokens:
                        eos_count = token_list.count(eos_token_id)
                        logger.warning(f"   Model generated eos_token_id ({eos_token_id}) {eos_count} times - generation may have ended immediately")
                    # Don't decode - will trigger empty text handling below
                    generated_text = ""
                else:
                    # Decode filtered tokens
                    logger.debug(f"Filtered {len(filtered_tokens)}/{num_new_tokens} tokens (removed {num_new_tokens - len(filtered_tokens)} special tokens)")
                    filtered_tensor = torch.tensor(filtered_tokens, device=generated_tokens.device)
                    
                    # Decode filtered tokens
                    generated_text = _global_tokenizer.decode(
                        filtered_tensor,
                        skip_special_tokens=True
                    ).strip()
                    logger.debug(f"Method 1 (filtered): Generated text length: {len(generated_text)} chars")
                
                # If still empty after filtering, try decoding original tokens with skip_special_tokens
                if not generated_text and len(filtered_tokens) > 0:
                    logger.debug("Filtered decoding produced empty text, trying original tokens with skip_special_tokens...")
                    generated_text = _global_tokenizer.decode(
                        generated_tokens,
                        skip_special_tokens=True
                    ).strip()
                    logger.debug(f"Method 1b (skip_special_tokens=True on original): Generated text length: {len(generated_text)} chars")
                
                # Debug: Show token IDs and what they decode to if still empty
                if not generated_text:
                    logger.warning(f"⚠️  All decoding methods produced empty text from {num_new_tokens} new tokens")
                    # Check if all tokens are pad/special tokens
                    if len(filtered_tokens) == 0:
                        logger.error(f"   ERROR: All {num_new_tokens} tokens are special tokens - model generated no content!")
                    # Show first 20 token IDs and their decoded values
                    sample_tokens = generated_tokens[:min(20, num_new_tokens)]
                    logger.debug("   First 20 token IDs and their decoded values:")
                    for i, tok_id in enumerate(sample_tokens):
                        try:
                            tok_id_val = tok_id.item()
                            tok_text = _global_tokenizer.decode([tok_id], skip_special_tokens=False)
                            is_special = tok_id_val in special_token_ids if special_token_ids else False
                            logger.debug(f"     Token {i}: ID={tok_id_val}, text='{tok_text[:50]}', is_special={is_special}")
                        except Exception as e:
                            logger.debug(f"     Token {i} (ID {tok_id.item()}): decode error - {e}")
                
                if generated_text:
                    logger.debug(f"Method 1 succeeded! First 100 chars: {generated_text[:100]}...")
                else:
                    logger.warning("Method 1 (token slicing) produced empty text after all attempts")
            
            # FALLBACK METHOD 2: Try exact string match if token slicing failed
            if not generated_text and full_text.startswith(prompt):
                logger.debug("Method 2 (exact match): Trying string-based stripping")
                generated_text = full_text[len(prompt):].strip()
                logger.debug(f"Method 2 (exact match): Generated text length: {len(generated_text)} chars")
            
            # FALLBACK METHOD 3: Try finding prompt in text
            if not generated_text:
                logger.debug("Method 3 (find prompt): Trying to find prompt in decoded text")
                prompt_index = full_text.find(prompt)
                if prompt_index >= 0:
                    generated_text = full_text[prompt_index + len(prompt):].strip()
                    logger.debug(f"Method 3 (find prompt): Found at index {prompt_index}, generated text length: {len(generated_text)} chars")
            
            # FALLBACK METHOD 4: If all else fails and we have more tokens than input, something is wrong
            if not generated_text:
                logger.error(f"⚠️ All methods failed to extract generated text!")
                logger.error(f"   Full text length: {len(full_text)}")
                logger.error(f"   Prompt length: {len(prompt)}")
                logger.error(f"   Output shape: {outputs.shape}")
                logger.error(f"   Input length: {input_length}")
                logger.error(f"   Expected new tokens: {outputs.shape[1] - input_length}")
                
                # Debug: Check what the new tokens decode to
                if outputs.shape[1] > input_length:
                    debug_tokens = outputs[0][input_length:input_length+10]  # First 10 new tokens
                    debug_text = _global_tokenizer.decode(debug_tokens, skip_special_tokens=True)
                    logger.error(f"   First 10 new tokens decode to: '{debug_text}'")
                
                # Last resort: Return full text minus a best-guess prompt length
                if len(full_text) > len(prompt):
                    generated_text = full_text[len(prompt):].strip()
                    logger.warning(f"Using last-resort fallback: returning text after prompt ({len(generated_text)} chars)")
                else:
                    raise RuntimeError(
                        f"Generated text is empty - model may not have generated any new tokens. "
                        f"Output has {outputs.shape[1]} tokens, input has {input_length} tokens, "
                        f"but decoded text ({len(full_text)} chars) equals prompt length ({len(prompt)} chars). "
                        f"This suggests the model repeated the prompt exactly."
                    )
            
            # Check if generated text is just pad tokens or special token strings
            if generated_text:
                # Check if text is all pad tokens (common issue)
                stripped_text = generated_text.strip()
                if stripped_text.replace('<pad>', '').strip() == '' or stripped_text.replace('<pad>', '').strip() == '':
                    logger.error(f"⚠️  Generated text is all pad tokens: '{generated_text[:100]}...'")
                    generated_text = ""  # Reset to empty so it triggers proper error handling
                elif '<pad>' in generated_text and len(stripped_text.replace('<pad>', '').strip()) < 10:
                    # Mostly pad tokens with a little actual content
                    logger.warning(f"⚠️  Generated text is mostly pad tokens, removing them...")
                    # Remove pad token strings
                    cleaned = generated_text.replace('<pad>', '').replace('<PAD>', '').replace('</s>', '').replace('<|endoftext|>', '').strip()
                    if cleaned and len(cleaned) > 10:
                        logger.info(f"   Cleaned pad tokens, remaining text: {len(cleaned)} chars")
                        generated_text = cleaned
                    else:
                        logger.warning("   After removing pad tokens, text is too short, treating as empty")
                        generated_text = ""
            
            logger.info(f"Inference complete. Generated text length: {len(generated_text)} characters")
            
            # Log a sample of the generated text for debugging (CRITICAL for troubleshooting)
            if generated_text:
                # Calculate statistics
                word_count = len(generated_text.split())
                char_count = len(generated_text)
                non_whitespace_count = len([c for c in generated_text if not c.isspace()])
                sentences = generated_text.count('.') + generated_text.count('!') + generated_text.count('?')
                
                logger.info(
                    f"Generated text stats: {char_count} chars, {word_count} words, "
                    f"{non_whitespace_count} non-whitespace chars, {sentences} sentences"
                )
                
                # Log samples at INFO level so they always show up
                sample = generated_text[:500].replace('\n', '\\n')
                logger.info(f"📝 Generated text sample (first 500 chars): {sample}")
                
                # Also log the last 200 chars to see how it ends
                if len(generated_text) > 500:
                    end_sample = generated_text[-200:].replace('\n', '\\n')
                    logger.info(f"📝 Generated text sample (last 200 chars): ...{end_sample}")
                
                # Check if text looks suspicious (mostly whitespace or special chars)
                if non_whitespace_count < char_count * 0.3:  # Less than 30% non-whitespace
                    logger.warning(
                        f"⚠️  Generated text has high whitespace ratio: "
                        f"{non_whitespace_count}/{char_count} ({non_whitespace_count/char_count*100:.1f}%) non-whitespace"
                    )
            else:
                logger.warning("⚠️  Generated text is empty after stripping!")
            
            return generated_text
            
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg:
                raise RuntimeError(
                    f"Out of memory error during generation. "
                    f"Try: (1) Reduce max_new_tokens (current: {max_new_tokens}), "
                    f"(2) Use CPU device, or (3) Reduce prompt length."
                ) from e
            else:
                raise RuntimeError(
                    f"Generation failed: {e}"
                ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error during generation: {e}"
            ) from e
