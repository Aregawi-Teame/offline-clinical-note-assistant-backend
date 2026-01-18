"""
Service for building prompts from templates and input data.
"""
from typing import Optional
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.generate import TaskEnum

logger = get_logger(__name__)


# Shared system instruction for medical documentation
SYSTEM_INSTRUCTION = """You are a medical documentation assistant. Your role is to generate accurate, professional clinical documentation based on the provided information.

KEY PRINCIPLES:
- Generate documentation that is factual, clear, and professional
- Follow the specified output format exactly
- Use medical terminology appropriately
- Maintain patient privacy by minimizing PHI where possible
- Be concise yet comprehensive
- Ensure accuracy and completeness within the provided information constraints"""


class PromptBuilder:
    """Build prompts using templates and input data."""
    
    # Map TaskEnum values to template filenames
    TASK_TO_TEMPLATE = {
        TaskEnum.SOAP: "soap.txt",
        TaskEnum.DISCHARGE: "discharge.txt",
        TaskEnum.REFERRAL: "referral.txt",
    }
    
    def __init__(self):
        """Initialize prompt builder with templates directory."""
        self.templates_dir = settings.PROMPT_DIR
        self._templates_cache: dict[str, str] = {}
    
    def _load_template(self, task: TaskEnum) -> str:
        """
        Load template file for a task type.
        
        Args:
            task: Task enum value (SOAP, DISCHARGE, REFERRAL)
            
        Returns:
            Template content as string
            
        Raises:
            ValueError: If template file is missing
        """
        # Check cache first
        task_key = task.value
        if task_key in self._templates_cache:
            return self._templates_cache[task_key]
        
        # Get template filename
        template_filename = self.TASK_TO_TEMPLATE.get(task)
        if not template_filename:
            raise ValueError(f"Unknown task type: {task}. Supported tasks: {list(TaskEnum)}")
        
        # Build template path
        template_path = Path(self.templates_dir) / template_filename
        
        # Check if template exists
        if not template_path.exists():
            raise ValueError(
                f"Template file not found: {template_path}. "
                f"Expected template for task '{task.value}' at {template_path.absolute()}"
            )
        
        # Load template
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read().strip()
            
            # Cache the template
            self._templates_cache[task_key] = template
            logger.debug(f"Loaded and cached template: {template_filename}")
            return template
            
        except IOError as e:
            raise ValueError(
                f"Error reading template file {template_path}: {e}. "
                f"Please ensure the file exists and is readable."
            ) from e
    
    def build(
        self,
        task: TaskEnum,
        notes: str,
        specialty: Optional[str] = None
    ) -> str:
        """
        Build a prompt from template and input data.
        
        Args:
            task: Task type (SOAP, DISCHARGE, REFERRAL)
            notes: Clinical notes/input data string
            specialty: Optional specialty (for referral tasks)
            
        Returns:
            Formatted prompt string with system instruction prepended
        """
        # Load template
        template = self._load_template(task)
        
        # Replace placeholders
        prompt = template.replace("{notes}", notes)
        
        # Handle specialty placeholder (only for referral)
        if task == TaskEnum.REFERRAL:
            # Replace specialty placeholder, use "Not specified" if not provided
            specialty_value = specialty if specialty else "Not specified"
            prompt = prompt.replace("{specialty}", specialty_value)
        elif "{specialty}" in prompt:
            # If specialty placeholder exists in non-referral template, remove it
            logger.warning(f"Specialty placeholder found in {task.value} template but task is not REFERRAL")
            prompt = prompt.replace("{specialty}", "Not applicable")
        
        # Prepend system instruction
        final_prompt = f"{SYSTEM_INSTRUCTION}\n\n{prompt}"
        
        logger.debug(
            f"Built prompt for {task.value}, "
            f"length: {len(final_prompt)}, "
            f"specialty: {specialty if specialty else 'N/A'}"
        )
        
        return final_prompt
