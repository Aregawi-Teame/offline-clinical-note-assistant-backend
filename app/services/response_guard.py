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
    # Social Security Numbers (SSN)
    (r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]'),  # 123-45-6789
    (r'\b\d{3}\s\d{2}\s\d{4}\b', '[REDACTED_SSN]'),  # 123 45 6789
    (r'\b\d{9}\b', '[REDACTED_SSN]'),  # 123456789 (9 consecutive digits - potential SSN)
    # Dates of Birth (DOB) - common formats
    (r'\b\d{1,2}/\d{1,2}/\d{4}\b', '[REDACTED_DOB]'),  # MM/DD/YYYY or M/D/YYYY
    (r'\b\d{4}-\d{2}-\d{2}\b', '[REDACTED_DOB]'),  # YYYY-MM-DD (ISO format)
    (r'\b\d{1,2}-\d{1,2}-\d{4}\b', '[REDACTED_DOB]'),  # MM-DD-YYYY or M-D-YYYY
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
        self._repair_attempted = False  # Prevent infinite recursion in format repair
    
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
        # Prevent infinite recursion: only attempt repair once per validation call
        if missing_sections and self.model_runner is not None and original_prompt and not self._repair_attempted:
            logger.warning(
                f"Missing required sections for {task.value}: {missing_sections}. "
                "Attempting format repair (one-time attempt)..."
            )
            self._repair_attempted = True  # Prevent recursion
            try:
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
            finally:
                self._repair_attempted = False  # Reset for next validation call
        elif missing_sections:
            if self._repair_attempted:
                logger.warning(f"Missing sections but repair already attempted, skipping to avoid recursion")
            else:
                logger.warning(
                    f"Missing required sections for {task.value}: {missing_sections}. "
                    "Repair unavailable (no model_runner provided)."
                )
        
        # Step 3.5: If sections are missing but we have meaningful content, 
        # log a warning but don't fail validation yet (will check at final validation)
        # This allows the response to pass if it has meaningful content even without perfect formatting
        
        # Step 5: Redact PHI patterns
        cleaned = self._redact_phi(cleaned)
        
        # Step 6: Final validation
        if len(cleaned) < self.min_length:
            logger.warning(f"Response too short after processing: {len(cleaned)} chars")
            raise ValueError(f"Response too short (minimum {self.min_length} characters)")
        
        # Check for meaningful content
        has_meaningful = self._has_meaningful_content(cleaned)
        
        # Final section check - be more lenient
        final_missing = self._check_required_sections(cleaned, task)
        
        # Log detailed info for debugging
        word_count = len(cleaned.split())
        char_count = len(cleaned)
        sentences = cleaned.count('.') + cleaned.count('!') + cleaned.count('?')
        
        logger.info(
            f"Validation check: {char_count} chars, {word_count} words, {sentences} sentences, "
            f"meaningful={has_meaningful}, missing_sections={final_missing}"
        )
        
        # Log a sample of the actual text to help debug
        sample = cleaned[:300].replace('\n', '\\n')
        logger.debug(f"Generated text sample (first 300 chars): {sample}")
        
        # If we have meaningful content, allow the response even if some sections are missing
        # This handles cases where the model generates valid clinical content but with different formatting
        if has_meaningful:
            if final_missing:
                # Log a warning but don't fail - the content is meaningful
                logger.warning(
                    f"Response has meaningful content ({len(cleaned)} chars) but missing sections: {final_missing}. "
                    "Allowing response with warning."
                )
                # Add a note at the beginning of the response about missing sections (informational only)
                section_note = f"[Note: Some sections may not be explicitly formatted: {', '.join(final_missing)}]\n\n"
                cleaned = section_note + cleaned
            logger.info(f"Validated response: {len(cleaned)} characters, task: {task.value}")
            return cleaned
        else:
            # For long responses (2000+ chars), be more lenient - accept if it has structure
            if char_count >= 2000:
                logger.warning(
                    f"Long response ({char_count} chars, {word_count} words) but keyword check failed. "
                    f"Allowing anyway if it has structure (sentences: {sentences})"
                )
                if sentences >= 3 or word_count >= 200:  # Has some structure
                    if final_missing:
                        logger.warning(f"Allowing long response despite missing sections: {final_missing}")
                        section_note = f"[Note: Some sections may not be explicitly formatted: {', '.join(final_missing)}]\n\n"
                        cleaned = section_note + cleaned
                    logger.info(f"Validated long response: {len(cleaned)} characters, task: {task.value}")
                    return cleaned
            
            # No meaningful content AND missing sections - fail validation
            # Log the actual text sample to help debug what was generated
            logger.error(
                f"Validation failed: {char_count} chars, {word_count} words, {sentences} sentences. "
                f"Missing sections: {final_missing}"
            )
            logger.error(f"Generated text sample (first 500 chars): {cleaned[:500]}")
            logger.error(f"Generated text sample (last 200 chars): {cleaned[-200:]}")
            
            raise ValueError(
                f"Response lacks meaningful content. Missing required sections: {final_missing} "
                f"(found {char_count} characters, {word_count} words, but no medical keywords detected). "
                f"Check logs for generated text sample."
            )
    
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
        found_sections = []
        missing = []
        
        for section in required:
            section_upper = section.upper()
            # Multiple patterns to catch different formatting styles:
            # 1. "SECTION:" or "SECTION :" (with colon)
            # 2. "SECTION" on its own line (line-start pattern)
            # 3. "**SECTION:**" (markdown bold)
            # 4. "# SECTION" (markdown header)
            patterns = [
                rf'\b{re.escape(section_upper)}\s*:',  # Standard: "SECTION:"
                rf'^{re.escape(section_upper)}\s*:?\s*$',  # Line-start: "SECTION:" at line start
                rf'^\s*#+\s*{re.escape(section_upper)}\s*$',  # Markdown: "# SECTION"
                rf'\*\*{re.escape(section_upper)}\*\*:?',  # Markdown bold: "**SECTION**"
                rf'<h[1-6]>\s*{re.escape(section_upper)}\s*</h[1-6]>',  # HTML header
                rf'\b{re.escape(section_upper)}\b',  # Just the word (as last resort)
            ]
            
            found = False
            for pattern in patterns:
                if re.search(pattern, text_upper, re.MULTILINE):
                    found = True
                    found_sections.append(section)
                    break
            
            if not found:
                missing.append(section)
        
        # Log what was found and what's missing
        if missing:
            logger.debug(
                f"Section header check for {task.value}: "
                f"Found {len(found_sections)}/{len(required)} sections. "
                f"Found: {found_sections}, Missing: {missing}"
            )
            # Log a sample of the actual text to help debug
            text_sample = text[:500].replace('\n', '\\n')
            logger.debug(f"Generated text sample (first 500 chars): {text_sample}")
        
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
        if not text or not text.strip():
            logger.debug("_has_meaningful_content: Empty or whitespace-only text")
            return False
        
        # Check for minimum word count (more lenient for long responses)
        word_count = len(text.split())
        char_count = len(text)
        
        if word_count < 5:
            logger.debug(f"_has_meaningful_content: Too few words ({word_count} < 5)")
            return False
        
        # For longer responses (200+ words OR 1000+ chars), be more lenient
        # Assume meaningful if it has sentence structure (indicates real text, not just tokens)
        if word_count >= 200 or char_count >= 1000:
            sentences = text.count('.') + text.count('!') + text.count('?')
            # Check for paragraph structure (multiple newlines)
            paragraphs = text.count('\n\n') + text.count('\r\n\r\n')
            
            # If it has sentences and paragraphs, it's likely meaningful
            if sentences >= 3:
                logger.debug(
                    f"Long response ({word_count} words, {char_count} chars, {sentences} sentences, "
                    f"{paragraphs} paragraphs) - assuming meaningful content based on structure"
                )
                return True
        
        # For medium responses (50-200 words), check for structure + keywords
        if word_count >= 50:
            sentences = text.count('.') + text.count('!') + text.count('?')
            if sentences >= 2:  # Has some sentence structure
                logger.debug(
                    f"Medium response ({word_count} words, {sentences} sentences) - "
                    "checking for keywords with lenient threshold"
                )
                # Only need 1 keyword for medium responses
                text_lower = text.lower()
                medical_keywords = [
                    "patient", "diagnosis", "treatment", "history", "examination",
                    "assessment", "plan", "subjective", "objective", "medication",
                    "discharge", "referral", "information", "provided", "clinical",
                    "symptom", "sign", "condition", "disease", "test", "result",
                    "vital", "sign", "physical", "chief", "complaint", "medical",
                    "care", "health", "doctor", "physician", "hospital", "visit"
                ]
                found_keywords = [kw for kw in medical_keywords if kw in text_lower]
                if len(found_keywords) >= 1:  # More lenient - only need 1 keyword
                    logger.debug(f"Found medical keyword(s): {found_keywords[:3]}... (total: {len(found_keywords)})")
                    return True
        
        # For short responses (10-50 words), be even more lenient
        # Short clinical notes like "Patient stable." are valid
        if word_count >= 10:
            text_lower = text.lower()
            # Accept if it has sentence structure OR contains any medical keyword
            sentences = text.count('.') + text.count('!') + text.count('?')
            if sentences >= 1:  # Has at least one sentence
                medical_keywords = [
                    "patient", "stable", "improved", "follow-up", "visit", "examination",
                    "assessment", "plan", "medication", "diagnosis", "treatment"
                ]
                found_keywords = [kw for kw in medical_keywords if kw in text_lower]
                if len(found_keywords) >= 1:  # Only need 1 keyword for short notes
                    logger.debug(f"Short response ({word_count} words) with sentence structure and keyword(s): {found_keywords}")
                    return True
        
        # Check for common medical note indicators (for shorter responses)
        medical_keywords = [
            "patient", "diagnosis", "treatment", "history", "examination",
            "assessment", "plan", "subjective", "objective", "medication",
            "discharge", "referral", "information", "provided", "clinical",
            "symptom", "sign", "condition", "disease", "test", "result",
            "vital", "sign", "physical", "chief", "complaint", "medical",
            "care", "health", "doctor", "physician", "hospital", "visit"
        ]
        
        # Check for common medical note indicators (for shortest responses)
        # Be lenient: only need 1 keyword for very short responses (5-10 words)
        # Examples: "Patient stable.", "Follow-up needed."
        text_lower = text.lower()
        found_keywords = [kw for kw in medical_keywords if kw in text_lower]
        
        # For very short responses (5-10 words), only need 1 keyword
        # For slightly longer (10+ words already handled above), need 2
        if word_count < 10:
            has_keywords = len(found_keywords) >= 1
            threshold = 1
        else:
            has_keywords = len(found_keywords) >= 2
            threshold = 2
        
        if has_keywords:
            logger.debug(f"Found medical keywords: {found_keywords[:5]}... (total: {len(found_keywords)}, needed: {threshold})")
        else:
            logger.debug(
                f"Medical keyword check failed: found {len(found_keywords)} keywords ({found_keywords[:3]}), "
                f"need at least {threshold}. Text sample: {text[:100]}"
            )
        
        return has_keywords
