"""
Service for validating and guarding model responses.
"""
import re
from typing import Optional

from app.core.logging import get_logger
from app.schemas.generate import TaskEnum

logger = get_logger(__name__)


# Required section headers for each task type
REQUIRED_SECTIONS = {
    TaskEnum.SOAP: ["SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN"],
    TaskEnum.DISCHARGE: [
        "ADMISSION INFORMATION",
        "HOSPITAL COURSE",
        "DISCHARGE INFORMATION",
        "DISCHARGE MEDICATIONS",
        "DISCHARGE INSTRUCTIONS",
        "FOLLOW-UP"
    ],
    TaskEnum.REFERRAL: [
        "REASON FOR REFERRAL",
        "RELEVANT HISTORY",
        "CURRENT MEDICATIONS",
        "CURRENT TREATMENTS",
        "RELEVANT DIAGNOSTICS",
        "SPECIFIC QUESTIONS OR CONCERNS",
        "URGENCY"
    ],
}

# PHI patterns for redaction
PHI_PATTERNS = [
    # Email addresses
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]'),
    # Phone numbers (various formats)
    (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[REDACTED_PHONE]'),  # 123-456-7890, 123.456.7890, 1234567890
    (r'\(\d{3}\)\s?\d{3}[-.]?\d{4}', '[REDACTED_PHONE]'),  # (123) 456-7890
    (r'\b\d{10}\b', '[REDACTED_PHONE]'),  # 10 consecutive digits
]


class ResponseGuard:
    """Validate and sanitize model responses."""
    
    MAX_OUTPUT_LENGTH = 6000
    
    def __init__(self, model_runner=None):
        """
        Initialize response guard.
        
        Args:
            model_runner: Optional ModelRunner instance for repair attempts
        """
        self.model_runner = model_runner
        self.min_length = 50
    
    def validate(
        self,
        response: str,
        task: TaskEnum,
        original_prompt: Optional[str] = None
    ) -> str:
        """
        Validate and guard model response.
        
        Args:
            response: Raw model response
            task: Task type to validate sections for
            original_prompt: Original prompt (used for repair attempts)
            
        Returns:
            Validated, sanitized, and safe response
            
        Raises:
            ValueError: If response fails critical validation
        """
        if not response or not isinstance(response, str):
            raise ValueError("Response must be a non-empty string")
        
        # Step 1: Basic cleaning
        cleaned = self._clean(response)
        
        # Step 2: Enforce max length
        if len(cleaned) > self.MAX_OUTPUT_LENGTH:
            logger.warning(f"Response exceeds max length: {len(cleaned)} chars, truncating to {self.MAX_OUTPUT_LENGTH}")
            cleaned = cleaned[:self.MAX_OUTPUT_LENGTH]
        
        # Step 3: Check for required section headers
        missing_sections = self._check_required_sections(cleaned, task)
        
        # Step 4: Attempt repair if sections are missing and model_runner is available
        if missing_sections and self.model_runner is not None and original_prompt:
            logger.warning(
                f"Missing required sections for {task.value}: {missing_sections}. "
                "Attempting format repair..."
            )
            repaired = self._attempt_format_repair(cleaned, original_prompt)
            if repaired:
                # Re-check sections after repair
                still_missing = self._check_required_sections(repaired, task)
                if not still_missing:
                    cleaned = repaired
                    logger.info("Format repair successful")
                else:
                    logger.warning(f"Format repair did not fix all sections. Still missing: {still_missing}")
            else:
                logger.warning("Format repair failed or unavailable")
        elif missing_sections:
            logger.warning(
                f"Missing required sections for {task.value}: {missing_sections}. "
                "Repair unavailable (no model_runner provided)."
            )
        
        # Step 5: Redact PHI patterns
        cleaned = self._redact_phi(cleaned)
        
        # Step 6: Final validation
        if len(cleaned) < self.min_length:
            logger.warning(f"Response too short after processing: {len(cleaned)} chars")
            raise ValueError(f"Response too short (minimum {self.min_length} characters)")
        
        if not self._has_meaningful_content(cleaned):
            raise ValueError("Response lacks meaningful content")
        
        logger.debug(f"Validated response: {len(cleaned)} characters, task: {task.value}")
        return cleaned
    
    def _clean(self, text: str) -> str:
        """
        Clean and normalize text response.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive blank lines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def _check_required_sections(self, text: str, task: TaskEnum) -> list[str]:
        """
        Check if response contains all required section headers.
        
        Args:
            text: Response text to check
            task: Task type
            
        Returns:
            List of missing section headers
        """
        required = REQUIRED_SECTIONS.get(task, [])
        if not required:
            return []
        
        text_upper = text.upper()
        missing = []
        
        for section in required:
            # Check if section header exists (case-insensitive)
            # Look for exact match or with colon
            pattern = rf'\b{re.escape(section.upper())}\s*:'
            if not re.search(pattern, text_upper):
                missing.append(section)
        
        return missing
    
    def _attempt_format_repair(
        self,
        response: str,
        original_prompt: str
    ) -> Optional[str]:
        """
        Attempt to repair format by re-running with repair instruction.
        
        Args:
            response: Current response with format issues
            original_prompt: Original prompt used
            
        Returns:
            Repaired response or None if repair fails
        """
        if not self.model_runner:
            return None
        
        try:
            # Create repair prompt
            repair_prompt = (
                f"{original_prompt}\n\n"
                "IMPORTANT: The previous response had formatting issues. "
                "Fix the format only - do not add any new facts or information. "
                "Ensure all required section headers are present and properly formatted. "
                "Use only the information already provided above.\n\n"
                f"Response to fix:\n{response[:2000]}"  # Limit context to avoid token limits
            )
            
            # Run repair with same options but lower temperature for more deterministic output
            repair_options = {
                "maxTokens": 1000,  # Limit tokens for repair
                "temperature": 0.1,  # Lower temperature for format-focused generation
                "topP": 0.9
            }
            
            repaired = self.model_runner.run(repair_prompt, repair_options)
            
            if repaired and len(repaired.strip()) > 50:
                return repaired.strip()
            
        except Exception as e:
            logger.error(f"Format repair failed: {e}")
        
        return None
    
    def _redact_phi(self, text: str) -> str:
        """
        Redact obvious PHI patterns from text.
        
        Args:
            text: Text to redact
            
        Returns:
            Text with PHI redacted
        """
        result = text
        
        for pattern, replacement in PHI_PATTERNS:
            matches = re.findall(pattern, result)
            if matches:
                logger.debug(f"Redacted {len(matches)} PHI pattern(s): {pattern[:30]}...")
                result = re.sub(pattern, replacement, result)
        
        return result
    
    def _has_meaningful_content(self, text: str) -> bool:
        """
        Check if text has meaningful content.
        
        Args:
            text: Text to check
            
        Returns:
            True if text has meaningful content
        """
        # Check for minimum word count
        word_count = len(text.split())
        if word_count < 10:
            return False
        
        # Check for common medical note indicators
        medical_keywords = [
            "patient", "diagnosis", "treatment", "history", "examination",
            "assessment", "plan", "subjective", "objective", "medication",
            "discharge", "referral", "information", "provided"
        ]
        
        text_lower = text.lower()
        has_keywords = sum(1 for keyword in medical_keywords if keyword in text_lower) >= 2
        
        return has_keywords
