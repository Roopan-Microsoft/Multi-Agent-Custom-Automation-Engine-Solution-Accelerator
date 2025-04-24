"""Define agent types for the Multi-Agent Custom Automation Engine."""

from enum import Enum


class AgentType(Enum):
    """Enum for agent types in the system."""
    
    HR = "hr_agent"
    MARKETING = "marketing_agent"
    PRODUCT = "product_agent" 
    PROCUREMENT = "procurement_agent"
    TECH_SUPPORT = "tech_support_agent"
    GENERIC = "generic_agent"
    HUMAN = "human_agent"
    PLANNER = "planner_agent"
    GROUP_CHAT_MANAGER = "group_chat_manager"
    
    def __str__(self) -> str:
        """Convert enum to string."""
        return self.value