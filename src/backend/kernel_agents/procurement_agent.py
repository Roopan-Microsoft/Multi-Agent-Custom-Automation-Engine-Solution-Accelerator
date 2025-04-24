from typing import List, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext

class ProcurementAgent(BaseAgent):
    """Procurement agent implementation using Semantic Kernel.
    
    This agent specializes in purchasing, vendor management, supply chain operations,
    and inventory control. It can create purchase orders, manage vendors, track orders,
    and ensure efficient procurement processes.
    """

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: Optional[List[KernelFunction]] = None,
        system_message: Optional[str] = None,
        agent_name: str = "ProcurementAgent",
        config_path: Optional[str] = None,
        client=None,
        definition=None,
    ) -> None:
        """Initialize the Procurement Agent.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: List of tools available to this agent (optional)
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "ProcurementAgent")
            config_path: Optional path to the Procurement tools configuration file
            client: Optional client instance
            definition: Optional definition instance
        """
        # Load configuration if tools not provided
        if tools is None:
            # Load the procurement tools configuration
            config = self.load_tools_config("procurement", config_path)
            tools = self.get_tools_from_config(kernel, "procurement", config_path)
            
            # Use system message from config if not explicitly provided
            if not system_message:
                system_message = config.get(
                    "system_message", 
                    "You are an AI Agent. You are able to assist with procurement enquiries and order items. If you need additional information from the human user asking the question in order to complete a request, ask before calling a function."
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