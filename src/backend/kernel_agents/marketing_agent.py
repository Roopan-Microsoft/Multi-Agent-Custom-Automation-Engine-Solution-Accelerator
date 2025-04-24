from typing import List, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext

class MarketingAgent(BaseAgent):
    """Marketing agent implementation using Semantic Kernel.
    
    This agent specializes in marketing strategies, campaign development,
    content creation, and market analysis. It can create effective marketing
    campaigns, analyze market trends, develop promotional content, and more.
    All tools are loaded from marketing_tools.json.
    """

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: Optional[List[KernelFunction]] = None,
        system_message: Optional[str] = None,
        agent_name: str = "MarketingAgent",
        config_path: Optional[str] = None,
        client=None,
        definition=None,
    ) -> None:
        """Initialize the Marketing Agent.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: List of tools available to this agent (optional)
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "MarketingAgent")
            config_path: Optional path to the Marketing tools configuration file
            client: Optional client instance
            definition: Optional definition instance
        """
        # Load configuration if tools not provided
        if tools is None:
            # Load the marketing tools configuration
            config = self.load_tools_config("marketing", config_path)
            tools = self.get_tools_from_config(kernel, "marketing", config_path)
            
            # Use system message from config if not explicitly provided
            if not system_message:
                system_message = config.get(
                    "system_message", 
                    "You are an AI Agent. You have knowledge about marketing, including campaigns, market research, and promotional activities."
                )
            
            # Use agent name from config if available
            agent_name = config.get("agent_name", agent_name)
        
        super().__init__(
            agent_name=agent_name,
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=tools,
            system_message=system_message,
            client=client,
            definition=definition
        )