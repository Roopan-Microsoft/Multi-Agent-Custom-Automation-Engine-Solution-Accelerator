from typing import List, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext

class TechSupportAgent(BaseAgent):
    """Tech Support agent implementation using Semantic Kernel.
    
    This agent specializes in IT troubleshooting, system administration, network issues,
    software installation, and general technical support. It can help with setting up software, 
    accounts, devices, and other IT-related tasks.
    """

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: Optional[List[KernelFunction]] = None,
        system_message: Optional[str] = None,
        agent_name: str = "TechSupportAgent",
        config_path: Optional[str] = None,
        client=None,
        definition=None,
    ) -> None:
        """Initialize the Tech Support Agent.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: List of tools available to this agent (optional)
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "TechSupportAgent")
            config_path: Optional path to the tech support tools configuration file
            client: Optional client instance
            definition: Optional definition instance
        """
        # Load configuration if tools not provided
        if tools is None:
            # Load the tech support tools configuration
            config = self.load_tools_config("tech_support", config_path)
            tools = self.get_tools_from_config(kernel, "tech_support", config_path)
            
            # Use system message from config if not explicitly provided
            if not system_message:
                system_message = config.get(
                    "system_message", 
                    "You are an AI Agent who is knowledgeable about Information Technology. You are able to help with setting up software, accounts, devices, and other IT-related tasks. If you need additional information from the human user asking the question in order to complete a request, ask before calling a function."
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