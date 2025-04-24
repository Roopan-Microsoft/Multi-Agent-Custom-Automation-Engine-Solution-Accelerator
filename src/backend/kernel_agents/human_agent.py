import logging
from typing import List, Optional

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.functions.kernel_arguments import KernelArguments

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import HumanFeedback, Step, StepStatus, AgentMessage, ActionRequest
from event_utils import track_event_if_configured

class HumanAgent(BaseAgent):
    """Human agent implementation using Semantic Kernel.
    
    This agent represents a human user in the system, receiving and processing
    feedback from humans and passing it to other agents for further action.
    """

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: Optional[List[KernelFunction]] = None,
        system_message: Optional[str] = None,
        agent_name: str = "HumanAgent",
        config_path: Optional[str] = None,
        client=None,
        definition=None,
    ) -> None:
        """Initialize the Human Agent.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: List of tools available to this agent (optional)
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "HumanAgent")
            config_path: Optional path to the Human tools configuration file
            client: Optional client instance
            definition: Optional definition instance
        """
        # Load configuration if tools not provided
        if tools is None:
            config = self.load_tools_config("human", config_path)
            tools = self.get_tools_from_config(kernel, "human", config_path)
            if not system_message:
                system_message = config.get(
                    "system_message", 
                    "You are representing a human user in the conversation. You handle interactions that require human feedback or input."
                )
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
        
    async def handle_human_feedback(self, kernel_arguments: KernelArguments) -> str:
        """Handle human feedback on a step.
        
        Args:
            kernel_arguments: Contains the human_feedback_json string
            
        Returns:
            Status message
        """
        # Parse the human feedback
        human_feedback_json = kernel_arguments["human_feedback_json"]
        human_feedback = HumanFeedback.parse_raw(human_feedback_json)
        
        # Get the step
        step = await self._memory_store.get_step(human_feedback.step_id, human_feedback.session_id)
        if not step:
            return f"Step {human_feedback.step_id} not found"
        
        # Update the step with the feedback
        step.human_feedback = human_feedback.human_feedback
        step.updated_action = human_feedback.updated_action
        
        if human_feedback.approved:
            step.status = StepStatus.approved
        else:
            step.status = StepStatus.needs_update
        
        # Save the updated step
        await self._memory_store.update_step(step)
        
        # If approved and updated action is provided, update the step's action
        if human_feedback.approved and human_feedback.updated_action:
            step.action = human_feedback.updated_action
            await self._memory_store.update_step(step)
        
        # Add a record of the feedback to the memory store
        await self._memory_store.add_item(
            AgentMessage(
                session_id=human_feedback.session_id,
                user_id=self._user_id,
                plan_id=step.plan_id,
                content=f"Received feedback for step: {step.action}",
                source="HumanAgent",
                step_id=human_feedback.step_id,
            )
        )
        
        # Track the event
        track_event_if_configured(
            f"Human Agent - Received feedback for step and added into the cosmos",
            {
                "session_id": human_feedback.session_id,
                "user_id": self._user_id,
                "plan_id": step.plan_id,
                "content": f"Received feedback for step: {step.action}",
                "source": "HumanAgent",
                "step_id": human_feedback.step_id,
            },
        )
        
        # Notify the GroupChatManager
        if human_feedback.approved:
            # Create a request to execute the next step
            group_chat_manager_id = f"group_chat_manager_{human_feedback.session_id}"
            
            # Use GroupChatManager's execute_next_step method
            if hasattr(self._kernel, 'get_service'):
                group_chat_manager = self._kernel.get_service(group_chat_manager_id)
                if group_chat_manager:
                    await group_chat_manager.execute_next_step(
                        KernelArguments(
                            session_id=human_feedback.session_id, 
                            plan_id=step.plan_id
                        )
                    )
                    
            # Track the approval request event
            track_event_if_configured(
                f"Human Agent - Approval request sent for step and added into the cosmos",
                {
                    "session_id": human_feedback.session_id,
                    "user_id": self._user_id,
                    "plan_id": step.plan_id,
                    "step_id": human_feedback.step_id,
                    "agent_id": "GroupChatManager",
                },
            )
        
        return "Human feedback processed successfully"
        
    async def provide_clarification(self, kernel_arguments: KernelArguments) -> str:
        """Provide clarification on a plan.
        
        Args:
            kernel_arguments: Contains session_id and clarification_text
            
        Returns:
            Status message
        """
        session_id = kernel_arguments["session_id"]
        clarification_text = kernel_arguments["clarification_text"]
        
        # Get the plan associated with this session
        plan = await self._memory_store.get_plan_by_session(session_id)
        if not plan:
            return f"No plan found for session {session_id}"
        
        # Update the plan with the clarification
        plan.human_clarification_response = clarification_text
        await self._memory_store.update_plan(plan)
        
        # Track the event
        track_event_if_configured(
            "Human Agent - Provided clarification for plan",
            {
                "session_id": session_id,
                "user_id": self._user_id,
                "plan_id": plan.id,
                "clarification": clarification_text,
            },
        )
        
        return f"Clarification provided for plan {plan.id}"