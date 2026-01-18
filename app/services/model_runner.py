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
        # Load tokenizer
        logger.debug("Loading tokenizer...")
        _global_tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load model
        logger.debug("Loading model (this may take a while)...")
        _global_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device in ("cuda", "mps") else torch.float32,
            device_map="auto" if device in ("cuda", "mps") else None,
            low_cpu_mem_usage=True,
        )
        
        # Move to device if not using device_map
        if device not in ("cuda", "mps") or not hasattr(_global_model, "hf_device_map"):
            _global_model = _global_model.to(device)
        
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
            inputs = _global_tokenizer(prompt, return_tensors="pt").to(self.device)
            input_length = inputs["input_ids"].shape[1]
            logger.debug(f"Tokenization complete. Input shape: {inputs['input_ids'].shape}, input_length={input_length}")
            
            # Prepare generation kwargs
            # For low temperatures (<= 0.2), use greedy decoding to avoid numerical issues
            # This is especially important for MPS and some GPU setups
            # Temperature 0.2 is default, so we use greedy by default (more stable)
            use_sampling = temperature > 0.2
            
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": _global_tokenizer.pad_token_id or _global_tokenizer.eos_token_id,
            }
            
            if use_sampling:
                # Ensure temperature is within safe range for sampling
                safe_temperature = max(0.3, min(temperature, 2.0))
                generation_kwargs["temperature"] = safe_temperature
                generation_kwargs["top_p"] = max(0.1, min(top_p, 1.0))
                generation_kwargs["do_sample"] = True
                logger.debug(f"Using sampling mode: temperature={safe_temperature}, top_p={generation_kwargs['top_p']}")
            else:
                # Greedy decoding for low/zero temperature (more stable, deterministic)
                # Don't pass temperature or top_p when using greedy decoding
                generation_kwargs["do_sample"] = False
                logger.debug("Using greedy decoding mode (temperature <= 0.2)")
            
            logger.info(f"Step 2: Starting model.generate() on device {self.device} with kwargs: {generation_kwargs}")
            logger.debug(f"Model device: {next(_global_model.parameters()).device}")
            logger.info(f"Input tensor device: {inputs['input_ids'].device}")
            
            # Device should already be resolved (MPS -> CPU) in _resolve_device
            # No need for fallback logic - device resolution handles MPS -> CPU conversion
            
            # Generate with torch.no_grad() for efficiency
            with torch.no_grad():
                logger.info(f"Step 3: Calling model.generate() on {self.device} - this may take a while...")
                outputs = _global_model.generate(
                    **inputs,
                    **generation_kwargs
                )
                logger.info(f"Generation complete. Output shape: {outputs.shape}")
            
            logger.debug("Step 4: Decoding output tokens...")
            # Decode the full output
            full_text = _global_tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug(f"Decoding complete. Full text length: {len(full_text)}")
            
            # Strip the prompt to return only generated text
            logger.info("Step 5: Stripping prompt from generated text...")
            if full_text.startswith(prompt):
                generated_text = full_text[len(prompt):].strip()
                logger.debug("Prompt found at start, stripped successfully")
            else:
                # If prompt doesn't match exactly, decode only the new tokens
                logger.debug("Prompt not at start, decoding only new tokens...")
                generated_tokens = outputs[0][input_length:]
                generated_text = _global_tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                ).strip()
            
            logger.info(f"Inference complete. Generated text length: {len(generated_text)} characters")
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
