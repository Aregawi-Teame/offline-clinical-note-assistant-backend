"""
Service for running the language model with Hugging Face Transformers.
"""
from typing import Optional
import time
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
        
        # Load model first so we can check actual embedding size
        logger.debug("Loading model (this may take a while)...")
        _global_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs
        )
        
        # CRITICAL FIX: Handle tokenizer/model embedding size mismatch
        # This is a common issue with MedGemma models where vocab_size != embedding_size
        embedding_size = _global_model.get_input_embeddings().num_embeddings
        tokenizer_vocab_size = _global_tokenizer.vocab_size
        logger.info(f"Model embedding size: {embedding_size}")
        logger.debug(f"Tokenizer vocab size: {tokenizer_vocab_size}")
        
        # CRITICAL FIX: If tokenizer vocab_size > embedding_size, we MUST resize the model
        # This prevents CUDA device-side assert errors from out-of-bounds token IDs
        if tokenizer_vocab_size > embedding_size:
            logger.error(f"⚠️  CRITICAL MISMATCH: tokenizer vocab_size ({tokenizer_vocab_size}) > model embedding_size ({embedding_size})")
            logger.error(f"   This WILL cause CUDA device-side assert errors!")
            logger.info(f"🔧 RESIZING model embeddings to match tokenizer vocab_size ({tokenizer_vocab_size})")
            
            # Resize the model embeddings to match tokenizer vocab_size
            _global_model.resize_token_embeddings(tokenizer_vocab_size)
            
            # Update embedding_size after resize
            embedding_size = _global_model.get_input_embeddings().num_embeddings
            logger.info(f"✅ Model embeddings resized to: {embedding_size}")
        
        # Warn if mismatch detected in other direction
        elif embedding_size != tokenizer_vocab_size:
            logger.warning(
                f"⚠️ MISMATCH DETECTED: tokenizer vocab_size ({tokenizer_vocab_size}) "
                f"!= model embedding_size ({embedding_size}). Model has extra capacity."
            )
        
        # Log current token IDs before modification
        logger.debug(
            f"Original token IDs - pad: {_global_tokenizer.pad_token_id}, "
            f"eos: {_global_tokenizer.eos_token_id}, "
            f"bos: {_global_tokenizer.bos_token_id}"
        )
        
        # CRITICAL FIX: Set pad_token_id if None (validate against EMBEDDING SIZE, not vocab_size)
        if _global_tokenizer.pad_token_id is None:
            if _global_tokenizer.eos_token_id is not None:
                # Validate eos_token_id is within EMBEDDING bounds (not tokenizer vocab)
                if _global_tokenizer.eos_token_id < embedding_size:
                    _global_tokenizer.pad_token_id = _global_tokenizer.eos_token_id
                    _global_tokenizer.pad_token = _global_tokenizer.eos_token
                    logger.info(f"Set pad_token_id to eos_token_id ({_global_tokenizer.eos_token_id}) - within embedding bounds")
                else:
                    # DON'T use vocab_size - 1! Add proper special token and resize embeddings
                    logger.error(f"eos_token_id ({_global_tokenizer.eos_token_id}) >= embedding_size ({embedding_size})")
                    logger.info("Adding new <pad> token and resizing model embeddings...")
                    _global_tokenizer.add_special_tokens({"pad_token": "<pad>"})
                    _global_model.resize_token_embeddings(len(_global_tokenizer))
                    _global_model.config.pad_token_id = _global_tokenizer.pad_token_id
                    logger.info(f"✅ Added pad token, resized embeddings to {len(_global_tokenizer)}")
            else:
                # No eos_token, add pad token properly
                logger.warning("No eos_token_id found, adding <pad> token...")
                _global_tokenizer.add_special_tokens({"pad_token": "<pad>"})
                _global_model.resize_token_embeddings(len(_global_tokenizer))
                _global_model.config.pad_token_id = _global_tokenizer.pad_token_id
                logger.info(f"✅ Added pad token, resized embeddings to {len(_global_tokenizer)}")
        else:
            # Validate existing pad_token_id is within embedding bounds
            if _global_tokenizer.pad_token_id >= embedding_size:
                logger.error(f"pad_token_id ({_global_tokenizer.pad_token_id}) >= embedding_size ({embedding_size})!")
                logger.info("Replacing invalid pad_token and resizing embeddings...")
                _global_tokenizer.add_special_tokens({"pad_token": "<pad>"})
                _global_model.resize_token_embeddings(len(_global_tokenizer))
                _global_model.config.pad_token_id = _global_tokenizer.pad_token_id
                logger.info(f"✅ Replaced invalid pad_token, resized embeddings to {len(_global_tokenizer)}")
            else:
                logger.debug(f"Existing pad_token_id ({_global_tokenizer.pad_token_id}) is valid")
        
        # CRITICAL FIX: Set padding side to "left" for CausalLM models
        _global_tokenizer.padding_side = "left"
        logger.debug(f"Set padding_side to 'left' for CausalLM compatibility")
        
        # CRITICAL: Hard validation - check ALL special tokens against embedding size
        logger.info("Validating all special tokens against model embedding size...")
        for token_name, token_id in {
            "eos": _global_tokenizer.eos_token_id,
            "pad": _global_tokenizer.pad_token_id,
            "bos": _global_tokenizer.bos_token_id,
        }.items():
            if token_id is not None:
                if token_id >= embedding_size:
                    logger.error(f"❌ CRITICAL: {token_name.upper()} token ID ({token_id}) >= "
                                   f"model embedding size ({embedding_size}). "
                                   f"This WILL cause CUDA device-side assert errors!")
                    # Fix: Clamp the token ID to valid range
                    fixed_token_id = min(token_id, embedding_size - 1)
                    logger.warning(f"🔧 Fixed {token_name}_token_id from {token_id} to {fixed_token_id}")
                    
                    # Update the tokenizer token ID
                    if token_name == "eos":
                        _global_tokenizer.eos_token_id = fixed_token_id
                    elif token_name == "pad":
                        _global_tokenizer.pad_token_id = fixed_token_id
                    elif token_name == "bos":
                        _global_tokenizer.bos_token_id = fixed_token_id
                else:
                    logger.debug(f"✅ {token_name}_token_id ({token_id}) is valid (< {embedding_size})")
        
        logger.info(f"✅ All special tokens validated successfully against embedding size ({embedding_size})")
        
        # Sync model config with tokenizer
        if hasattr(_global_model, 'config') and _global_tokenizer.pad_token_id is not None:
            _global_model.config.pad_token_id = _global_tokenizer.pad_token_id
            logger.debug(f"Set model.config.pad_token_id to {_global_tokenizer.pad_token_id}")
        
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
        
        # Handle both camelCase (from API) and snake_case formats
        max_new_tokens = options.get("max_new_tokens") or options.get("maxTokens", self.default_max_new_tokens)
        temperature = options.get("temperature", self.default_temperature)
        top_p = options.get("top_p") or options.get("topP", self.default_top_p)
        
        logger.info(
            f"Starting inference: prompt_len={len(prompt)}, "
            f"max_new_tokens={max_new_tokens}, "
            f"temperature={temperature}, top_p={top_p}, "
            f"device={self.device}"
        )
        
        try:
            # CRITICAL FIX: Comprehensive input validation BEFORE tokenization
            # This prevents CUDA device-side assert errors from malformed input
            logger.debug(f"Step 0: Pre-tokenization validation - prompt length: {len(prompt)} chars")
            
            # Check for potential issues in prompt that could cause tokenization problems
            if len(prompt.strip()) == 0:
                raise ValueError("Empty prompt provided")
            
            if len(prompt) > 50000:  # Reasonable limit to prevent OOM
                logger.warning(f"Very long prompt ({len(prompt)} chars) may cause issues")
            
            # Tokenize input with enhanced validation
            logger.debug("Step 1: Tokenizing input prompt...")
            try:
                # Tokenize and move to device
                # Note: attention_mask is automatically included in inputs dict
                inputs = _global_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                input_length = inputs["input_ids"].shape[1]
                
                # CRITICAL: Validate tokenization results
                if input_length == 0:
                    raise ValueError("Tokenization produced empty input")
                
                if input_length > 2048:  # Conservative limit for MedGemma
                    logger.warning(f"Long input sequence ({input_length} tokens) may cause CUDA issues")
                
            except Exception as tokenize_error:
                logger.error(f"Tokenization failed: {tokenize_error}")
                raise RuntimeError(f"Failed to tokenize prompt: {tokenize_error}") from tokenize_error
            
            # Log tokenization details for debugging BOS/EOS token issues
            # Check if BOS token was added (some tokenizers add it, some don't)
            if hasattr(_global_tokenizer, 'bos_token_id') and _global_tokenizer.bos_token_id is not None:
                first_token_id = inputs["input_ids"][0, 0].item()
                if first_token_id == _global_tokenizer.bos_token_id:
                    logger.debug(f"BOS token detected at start of input (token_id: {first_token_id})")
            
            logger.debug(f"Tokenization complete. Input shape: {inputs['input_ids'].shape}, input_length={input_length}")
            logger.debug(f"Attention mask shape: {inputs.get('attention_mask', 'not provided').shape if hasattr(inputs.get('attention_mask', None), 'shape') else 'not provided'}")
            
            # Prepare generation kwargs
            # CRITICAL FIX: Enhanced token validation with comprehensive bounds checking
            # This prevents CUDA device-side assert errors from out-of-bounds token IDs
            embedding_size = _global_model.get_input_embeddings().num_embeddings
            input_token_ids = inputs["input_ids"][0]
            max_input_token_id = input_token_ids.max().item()
            min_input_token_id = input_token_ids.min().item()
            
            logger.debug(f"Input token ID range: [{min_input_token_id}, {max_input_token_id}], embedding_size: {embedding_size}")
            
            # CRITICAL: Check if any input tokens are out of bounds
            if max_input_token_id >= embedding_size:
                logger.error(f"⚠️  INPUT TOKENS OUT OF BOUNDS: max token ID ({max_input_token_id}) >= embedding_size ({embedding_size})")
                logger.error(f"   This WILL cause CUDA device-side assert errors!")
                
                # EMERGENCY FIX: Clamp out-of-bounds tokens to valid range
                # This is a last resort to prevent CUDA crashes
                unk_token_id = _global_tokenizer.unk_token_id if _global_tokenizer.unk_token_id is not None else _global_tokenizer.pad_token_id
                if unk_token_id is None or unk_token_id >= embedding_size:
                    # Fallback: use 0 as safe token ID (usually BOS or safe range)
                    unk_token_id = 0
                    if unk_token_id >= embedding_size:
                        unk_token_id = embedding_size - 1  # Last valid token
                        logger.warning(f"Using last valid token ({unk_token_id}) as replacement")
                
                # Clamp the input tokens
                inputs["input_ids"] = torch.clamp(inputs["input_ids"], 0, embedding_size - 1)
                logger.warning(f"🔧 CLAMPED input tokens to range [0, {embedding_size - 1}]")
                logger.info(f"   Original max token: {max_input_token_id}, clamped max: {inputs['input_ids'].max().item()}")
                
                # Also update attention mask if it exists
                if "attention_mask" in inputs:
                    # Ensure attention mask matches clamped input_ids
                    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
                    logger.debug("Updated attention mask to match clamped input_ids")
            
            # Get token IDs (validate they're within bounds)
            eos_token_id = _global_tokenizer.eos_token_id
            pad_token_id = _global_tokenizer.pad_token_id
            bos_token_id = _global_tokenizer.bos_token_id
            unk_token_id = _global_tokenizer.unk_token_id
            
            # CRITICAL: Validate ALL special token IDs before passing to model
            special_tokens = {
                "eos": eos_token_id,
                "pad": pad_token_id, 
                "bos": bos_token_id,
                "unk": unk_token_id
            }
            
            for token_name, token_id in special_tokens.items():
                if token_id is not None and token_id >= embedding_size:
                    logger.error(f"⚠️  {token_name.upper()}_token_id ({token_id}) >= embedding_size ({embedding_size}) - this will cause CUDA errors!")
                    # Fix: Clamp the token ID to valid range
                    fixed_token_id = min(token_id, embedding_size - 1)
                    logger.warning(f"🔧 Fixed {token_name}_token_id from {token_id} to {fixed_token_id}")
                    
                    # Update the tokenizer token ID
                    if token_name == "eos":
                        _global_tokenizer.eos_token_id = fixed_token_id
                        eos_token_id = fixed_token_id
                    elif token_name == "pad":
                        _global_tokenizer.pad_token_id = fixed_token_id
                        pad_token_id = fixed_token_id
                    elif token_name == "bos":
                        _global_tokenizer.bos_token_id = fixed_token_id
                        bos_token_id = fixed_token_id
                    elif token_name == "unk":
                        _global_tokenizer.unk_token_id = fixed_token_id
                        unk_token_id = fixed_token_id
                else:
                    logger.debug(f"✅ {token_name}_token_id ({token_id}) is valid (< {embedding_size})")
            
            logger.debug(f"Token IDs validated: eos={eos_token_id}, pad={pad_token_id}, embedding_size={embedding_size}")
            
            # Determine if we should use sampling
            # CRITICAL FIX: If top_p is provided (> 0), use sampling to avoid getting stuck in loops
            # Greedy decoding can get stuck in repetitive pad token loops
            use_sampling = temperature > 0.2 or top_p > 0
            
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": 1.1,  # Slight penalty to reduce repetition loops
            }
            
            # Only add eos_token_id if it's valid (not None and within bounds)
            if eos_token_id is not None:
                generation_kwargs["eos_token_id"] = eos_token_id
                logger.debug(f"Set eos_token_id={eos_token_id} in generation kwargs")
            
            # IMPORTANT: Do NOT pass pad_token_id to generation kwargs
            # Let the model use its default padding behavior
            # Explicitly setting pad_token_id can cause generation to produce pad tokens instead of content
            
            if use_sampling:
                # Ensure temperature is within safe range for sampling
                safe_temperature = max(0.3, min(temperature, 2.0))
                generation_kwargs["temperature"] = safe_temperature
                generation_kwargs["top_p"] = max(0.1, min(top_p, 1.0))
                generation_kwargs["do_sample"] = True
                logger.debug(
                    f"Using sampling mode: temperature={safe_temperature}, top_p={generation_kwargs['top_p']}, "
                    f"repetition_penalty={generation_kwargs['repetition_penalty']}"
                )
            else:
                # Greedy decoding for low/zero temperature
                # Note: Greedy decoding can get stuck in loops, but is more deterministic
                generation_kwargs["do_sample"] = False
                logger.debug(
                    f"Using greedy decoding mode (temperature={temperature}, no top_p), "
                    f"repetition_penalty={generation_kwargs['repetition_penalty']}"
                )
            
            logger.debug(f"Generation kwargs prepared: {generation_kwargs}")
            logger.info(f"Step 2: Starting model.generate() on device {self.device} with kwargs: {generation_kwargs}")
            logger.debug(f"Model device: {next(_global_model.parameters()).device}")
            logger.info(f"Input tensor device: {inputs['input_ids'].device}")
            logger.info(f"⏳ Generation may take 2-5 minutes for {max_new_tokens} tokens on T4 GPU. Please wait...")
            
            # Device should already be resolved (MPS -> CPU) in _resolve_device
            # No need for fallback logic - device resolution handles MPS -> CPU conversion
            
            # CRITICAL FIX: Add CPU fallback for persistent CUDA errors
            # This ensures the service remains available even if CUDA has issues
            generation_start = time.time()
            with torch.no_grad():
                logger.info(f"Step 3: Calling model.generate() on {self.device} - this may take a while...")
                try:
                    # CRITICAL FIX: Explicitly pass attention_mask to ensure model "sees" the prompt correctly
                    # This prevents the model from failing to see the prompt and defaulting to pad tokens
                    outputs = _global_model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),  # Explicitly pass attention mask
                        **generation_kwargs
                    )
                    generation_time = time.time() - generation_start
                    logger.info(f"Generation complete in {generation_time:.1f}s. Output shape: {outputs.shape}")
                except RuntimeError as cuda_error:
                    generation_time = time.time() - generation_start
                    error_msg = str(cuda_error).lower()
                    
                    # Check if it's a CUDA device-side assert error
                    if "device-side assert" in error_msg or "cuda error" in error_msg:
                        logger.error(f"CUDA device-side assert detected after {generation_time:.1f}s: {cuda_error}")
                        
                        # If we're on CUDA and this is a device-side assert, try CPU fallback
                        if self.device != "cpu":
                            logger.warning("🔄 Attempting CPU fallback due to CUDA device-side assert...")
                            try:
                                # Move inputs to CPU
                                cpu_inputs = {k: v.cpu() for k, v in inputs.items()}
                                
                                # Try generation on CPU
                                outputs = _global_model.cpu().generate(
                                    input_ids=cpu_inputs["input_ids"],
                                    attention_mask=cpu_inputs.get("attention_mask"),
                                    **generation_kwargs
                                )
                                
                                # Move outputs back to original device for consistency
                                outputs = outputs.to(self.device)
                                generation_time = time.time() - generation_start
                                logger.info(f"✅ CPU fallback successful! Generation completed in {generation_time:.1f}s")
                                logger.warning("⚠️  Consider using DEVICE=cpu permanently if CUDA errors persist")
                                
                            except Exception as cpu_fallback_error:
                                logger.error(f"❌ CPU fallback also failed: {cpu_fallback_error}")
                                raise RuntimeError(
                                    f"CUDA device-side assert error and CPU fallback both failed. "
                                    f"CUDA error: {cuda_error}. CPU error: {cpu_fallback_error}. "
                                    f"Try using DEVICE=cpu or reducing prompt length."
                                ) from cuda_error
                        else:
                            # We're already on CPU and still getting errors
                            raise RuntimeError(
                                f"CUDA device-side assert error on CPU: {cuda_error}. "
                                f"This suggests a model/tokenizer compatibility issue. "
                                f"Try using a different model or checking prompt content."
                            ) from cuda_error
                    else:
                        # Different type of RuntimeError
                        raise cuda_error
                        
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
