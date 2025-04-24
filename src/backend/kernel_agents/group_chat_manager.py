import logging
import json
from datetime import datetime
import re
from typing import Dict, List, Optional, Any, Tuple

import semantic_kernel as sk
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.agents import AgentGroupChat  # pylint: disable=E0611

from semantic_kernel.agents.strategies import (
    SequentialSelectionStrategy,
    TerminationStrategy,
)
# Updated imports for compatibility 
try:
    # Try importing from newer structure first
    from semantic_kernel.contents import ChatMessageContent, ChatHistory
except ImportError:
    # Fall back to older structure for compatibility
    class ChatMessageContent:
        """Compatibility class for older SK versions."""
        def __init__(self, role="", content="", name=None):
            self.role = role
            self.content = content
            self.name = name

    class ChatHistory:
        """Compatibility class for older SK versions."""
        def __init__(self):
            self.messages = []

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import (
    ActionRequest,
    ActionResponse,
    AgentMessage,
    Step,
    StepStatus,
    PlanStatus,
    HumanFeedbackStatus,
    InputTask,
    Plan,
)
from models.agent_types import AgentType
from event_utils import track_event_if_configured


class GroupChatManagerClass:
    """A class for service compatibility with Semantic Kernel."""
    # Defining properties needed by Semantic Kernel
    service_id = ""
    
    def __init__(self, manager):
        self.manager = manager
        self.service_id = f"group_chat_manager_{manager._session_id}"
    
    async def execute_next_step(self, kernel_arguments: KernelArguments) -> str:
        """Execute the next step in the plan.
        
        Args:
            kernel_arguments: KernelArguments that should contain session_id and plan_id
            
        Returns:
            Status message
        """
        return await self.manager.execute_next_step(kernel_arguments)


class GroupChatManager:
    """Group Chat Manager implementation using Semantic Kernel's AgentGroupChat.
    
    This manager coordinates conversations between different agents and ensures
    the plan executes smoothly by orchestrating agent interactions.
    """

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        available_agents: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the Group Chat Manager.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            config_path: Optional path to the group_chat_manager tools configuration file
            available_agents: Dictionary of available agents mapped by their name
        """
        self._kernel = kernel
        self._session_id = session_id
        self._user_id = user_id
        self._memory_store = memory_store
        
        # Store available agents
        self._agent_instances = available_agents or {}
        
        # Initialize the AgentGroupChat later when all agents are registered
        self._agent_group_chat = None
        self._initialized = False
        
        # Create a wrapper class for service registration
        service_wrapper = GroupChatManagerClass(self)
        
        try:
            # Register with kernel using the service_wrapper
            if hasattr(kernel, "register_services"):
                kernel.register_services({service_wrapper.service_id: service_wrapper})
                logging.info(f"Registered GroupChatManager as kernel service with ID: {service_wrapper.service_id}")
            elif hasattr(kernel, "services") and hasattr(kernel.services, "register_service"):
                kernel.services.register_service(service_wrapper.service_id, service_wrapper)
                logging.info(f"Registered GroupChatManager as kernel service with ID: {service_wrapper.service_id}")
            elif hasattr(kernel, "services") and isinstance(kernel.services, dict):
                # Last resort: directly add to services dictionary
                kernel.services[service_wrapper.service_id] = service_wrapper
                logging.info(f"Added GroupChatManager to kernel services dictionary with ID: {service_wrapper.service_id}")
            else:
                logging.warning("Could not register GroupChatManager service. Semantic Kernel version might be incompatible.")
        except Exception as e:
            logging.error(f"Error registering GroupChatManager service: {e}")
            # Continue without crashing

    async def initialize_group_chat(self) -> None:
        """Initialize the AgentGroupChat with registered agents and strategies."""
        if self._initialized:
            return
            
        # Create the AgentGroupChat with registered agents and strategies
        self._agent_group_chat = AgentGroupChat(
            agents=list(self._agent_instances.values()),
            termination_strategy=self.PlanTerminationStrategy(agents=list(self._agent_instances.values())),
            selection_strategy=self.PlanSelectionStrategy(agents=list(self._agent_instances.values())),
        )
        
        self._initialized = True
        logging.info(f"Initialized AgentGroupChat with {len(self._agent_instances)} agents")

    async def register_agent(self, agent_name: str, agent: BaseAgent) -> None:
        """Register an agent with the Group Chat Manager.
        
        Args:
            agent_name: The name of the agent
            agent: The agent instance
        """
        self._agent_instances[agent_name] = agent
        self._initialized = False  # Need to re-initialize after adding new agents
        logging.info(f"Registered agent {agent_name} with Group Chat Manager")

    class PlanSelectionStrategy(SequentialSelectionStrategy):
        """Strategy for determining which agent should take the next turn in the chat.
        
        This strategy follows the progression of a plan, selecting agents based on
        the current step or phase of the plan execution.
        """

        async def select_agent(self, agents, history):
            """Select the next agent that should take the turn in the chat.
            
            Args:
                agents: List of available agents
                history: Chat history (ChatHistory object)
                
            Returns:
                The next agent to take the turn
            """
            # If no history, start with the PlannerAgent
            if not history or not history.messages:
                return next((agent for agent in agents if agent.name == "PlannerAgent"), None)
            
            # Get the last message
            last_message = history.messages[-1]
            
            # Extract name from the message - in SK ChatMessageContent
            last_sender = last_message.role
            if hasattr(last_message, 'name') and last_message.name:
                last_sender = last_message.name
            
            # Route based on the last sender
            if last_sender == "PlannerAgent":
                # After the planner creates a plan, HumanAgent should review it
                return next((agent for agent in agents if agent.name == "HumanAgent"), None)
            elif last_sender == "HumanAgent":
                # After human feedback, the specific agent for the step should proceed
                # For simplicity, use GenericAgent as fallback
                return next((agent for agent in agents if agent.name == "GenericAgent"), None)
            elif last_sender == "GroupChatManager":
                # If the manager just assigned a step, find the agent that should execute it
                # For simplicity, just rotate to the next agent
                agent_names = [agent.name for agent in agents]
                try:
                    current_index = agent_names.index(last_sender)
                    next_index = (current_index + 1) % len(agents)
                    return agents[next_index]
                except ValueError:
                    return agents[0] if agents else None
            else:
                # Default to the Group Chat Manager
                return next((agent for agent in agents if agent.name == "GroupChatManager"), 
                           agents[0] if agents else None)

    class PlanTerminationStrategy(TerminationStrategy):
        """Strategy for determining when the agent group chat should terminate.
        
        This strategy decides when the plan is complete or when a human needs to
        provide additional input to continue.
        """
        
        def __init__(self, agents, maximum_iterations=10, automatic_reset=True):
            """Initialize the termination strategy.
            
            Args:
                agents: List of agents in the group chat
                maximum_iterations: Maximum number of iterations before termination
                automatic_reset: Whether to reset the agent after termination
            """
            super().__init__(maximum_iterations, automatic_reset)
            self._agents = agents
        
        async def should_terminate(self, history, agents=None) -> bool:
            """Check if the chat should terminate.
            
            Args:
                history: Chat history as a ChatHistory object
                agents: List of agents (optional, uses self._agents if not provided)
                
            Returns:
                True if the chat should terminate, False otherwise
            """
            # Default termination conditions from parent class
            if await super().should_terminate(history, agents or self._agents):
                return True
                
            # If no history, continue the chat
            if not history or not history.messages:
                return False
            
            # Get the last message
            last_message = history.messages[-1]
            
            # End the chat if the plan is completed or if human intervention is required
            if "plan completed" in last_message.content.lower():
                return True
                
            if "human intervention required" in last_message.content.lower():
                return True
                
            # Terminate if we encounter a specific error condition
            if "error" in last_message.content.lower() and "cannot proceed" in last_message.content.lower():
                return True
                
            # Otherwise, continue the chat
            return False

    async def handle_input_task(self, input_task_json: str) -> str:
        """Handle the initial input task from the user.
        
        Args:
            input_task_json: Input task in JSON format
            
        Returns:
            Status message
        """
        # Parse the input task
        input_task = InputTask.parse_raw(input_task_json)
        
        # Store the user's message
        await self._memory_store.add_item(
            AgentMessage(
                session_id=input_task.session_id,
                user_id=self._user_id,
                plan_id="",
                content=f"{input_task.description}",
                source="HumanAgent",
                step_id="",
            )
        )
        
        track_event_if_configured(
            "Group Chat Manager - Received and added input task into the cosmos",
            {
                "session_id": input_task.session_id,
                "user_id": self._user_id,
                "content": input_task.description,
                "source": "HumanAgent",
            },
        )
        
        # Ensure the planner agent is registered
        if "PlannerAgent" not in self._agent_instances:
            return "PlannerAgent not registered. Cannot create plan."
            
        # Get the planner agent
        planner_agent = self._agent_instances["PlannerAgent"]
        
        # Forward the input task to the planner agent to create a plan
        planner_args = KernelArguments(input_task_json=input_task_json)
        plan_result = await planner_agent.handle_input_task(planner_args)
        
        return f"Plan creation initiated: {plan_result}"
    
    async def handle_human_feedback(self, human_feedback_json: str) -> str:
        """Handle human feedback on steps.
        
        Args:
            human_feedback_json: Human feedback in JSON format
            
        Returns:
            Status message
        """
        # Parse the human feedback
        human_feedback = json.loads(human_feedback_json)
        
        session_id = human_feedback.get("session_id", "")
        plan_id = human_feedback.get("plan_id", "")
        step_id = human_feedback.get("step_id", "")
        approved = human_feedback.get("approved", False)
        feedback_text = human_feedback.get("human_feedback", "")
        
        # Get general information
        general_information = f"Today's date is {datetime.now().date()}."
        
        # Get the plan
        plan = await self._memory_store.get_plan(plan_id)
        if not plan:
            return f"Plan {plan_id} not found"
            
        # Get plan human clarification if available
        if hasattr(plan, 'human_clarification_response') and plan.human_clarification_response:
            received_human_feedback_on_plan = (
                plan.human_clarification_response
                + " This information may or may not be relevant to the step you are executing - it was feedback provided by the human user on the overall plan, which includes multiple steps, not just the one you are actioning now."
            )
        else:
            received_human_feedback_on_plan = "No human feedback provided on the overall plan."
            
        # Combine all feedback into a single string
        received_human_feedback = (
            f"{feedback_text} "
            f"{general_information} "
            f"{received_human_feedback_on_plan}"
        )
        
        # Get all steps for the plan
        steps = await self._memory_store.get_steps_for_plan(plan_id, session_id)
        
        # Update and execute the specific step if step_id is provided
        if step_id:
            step = next((s for s in steps if s.id == step_id), None)
            if step:
                await self._update_step_status(step, approved, received_human_feedback)
                if approved:
                    return await self._execute_step(session_id, step)
                else:
                    # Handle rejected step
                    step.status = StepStatus.rejected
                    if hasattr(step, 'human_approval_status'):
                        step.human_approval_status = HumanFeedbackStatus.rejected
                    await self._memory_store.update_step(step)
                    
                    track_event_if_configured(
                        "Group Chat Manager - Step has been rejected and updated into the cosmos",
                        {
                            "status": StepStatus.rejected,
                            "session_id": session_id,
                            "user_id": self._user_id,
                            "human_approval_status": "rejected",
                            "source": step.agent,
                        },
                    )
                    return f"Step {step_id} rejected"
            else:
                return f"Step {step_id} not found"
        else:
            # Update all steps if no specific step_id is provided
            updates_count = 0
            for step in steps:
                if step.status == StepStatus.planned:
                    await self._update_step_status(step, approved, received_human_feedback)
                    if approved:
                        await self._execute_step(session_id, step)
                    updates_count += 1
            
            return f"Updated {updates_count} steps with human feedback"
            
    async def _update_step_status(self, step: Step, approved: bool, received_human_feedback: str) -> None:
        """Update a step's status based on human feedback.
        
        Args:
            step: The step to update
            approved: Whether the step is approved
            received_human_feedback: Feedback from human
        """
        if approved:
            step.status = StepStatus.approved
            if hasattr(step, 'human_approval_status'):
                step.human_approval_status = HumanFeedbackStatus.accepted
        else:
            step.status = StepStatus.rejected
            if hasattr(step, 'human_approval_status'):
                step.human_approval_status = HumanFeedbackStatus.rejected
                
        step.human_feedback = received_human_feedback
        await self._memory_store.update_step(step)
        
        track_event_if_configured(
            "Group Chat Manager - Received human feedback, Updating step and updated into the cosmos",
            {
                "status": step.status,
                "session_id": step.session_id,
                "user_id": self._user_id,
                "human_feedback": received_human_feedback,
                "source": step.agent,
            },
        )
        
    async def _execute_step(self, session_id: str, step: Step) -> str:
        """Execute a step by sending an action request to the appropriate agent.
        
        Args:
            session_id: The session identifier
            step: The step to execute
            
        Returns:
            Status message
        """
        # Update step status
        step.status = StepStatus.action_requested
        await self._memory_store.update_step(step)
        
        track_event_if_configured(
            "Group Chat Manager - Update step to action_requested and updated into the cosmos",
            {
                "status": StepStatus.action_requested,
                "session_id": step.session_id,
                "user_id": self._user_id,
                "source": step.agent,
            },
        )
        
        # Generate conversation history for context
        plan = await self._memory_store.get_plan(step.plan_id)
        steps = await self._memory_store.get_steps_for_plan(step.plan_id, session_id)
        conversation_history = await self._generate_conversation_history(steps, step.id, plan)
        
        # Create action request with conversation history for context
        action_with_history = f"{conversation_history} Here is the step to action: {step.action}. ONLY perform the steps and actions required to complete this specific step, the other steps have already been completed. Only use the conversational history for additional information, if it's required to complete the step you have been assigned."
        
        # Format agent name for display
        if hasattr(step, 'agent') and step.agent:
            agent_name = step.agent
            formatted_agent = re.sub(r"([a-z])([A-Z])", r"\1 \2", agent_name)
        else:
            # Default to GenericAgent if none specified
            agent_name = "GenericAgent"
            formatted_agent = "Generic Agent"
        
        # Store the agent message
        await self._memory_store.add_item(
            AgentMessage(
                session_id=session_id,
                user_id=self._user_id,
                plan_id=step.plan_id,
                content=f"Requesting {formatted_agent} to perform action: {step.action}",
                source="GroupChatManager",
                step_id=step.id,
            )
        )
        
        track_event_if_configured(
            f"Group Chat Manager - Requesting {agent_name} to perform the action and added into the cosmos",
            {
                "session_id": session_id,
                "user_id": self._user_id,
                "plan_id": step.plan_id,
                "content": f"Requesting {agent_name} to perform action: {step.action}",
                "source": "GroupChatManager",
                "step_id": step.id,
            },
        )
        
        # Special handling for HumanAgent
        if agent_name == "HumanAgent":
            # Mark as completed since we have received the human feedback
            step.status = StepStatus.completed
            await self._memory_store.update_step(step)
            
            logging.info("Marking the step as complete - Since we have received the human feedback")
            track_event_if_configured(
                "Group Chat Manager - Steps completed - Received the human feedback and updated into the cosmos",
                {
                    "session_id": session_id,
                    "user_id": self._user_id,
                    "plan_id": step.plan_id,
                    "content": "Marking the step as complete - Since we have received the human feedback",
                    "source": agent_name,
                    "step_id": step.id,
                },
            )
            return f"Step {step.id} for HumanAgent marked as completed"
        
        # Check if agent is registered
        if agent_name not in self._agent_instances:
            logging.warning(f"Agent {agent_name} not found. Using GenericAgent instead.")
            agent_name = "GenericAgent"
            if agent_name not in self._agent_instances:
                return f"No agent found to handle step {step.id}"
        
        # Create action request
        action_request = ActionRequest(
            step_id=step.id,
            plan_id=step.plan_id,
            session_id=session_id,
            action=action_with_history
        )
        
        # Send action request to the agent
        agent = self._agent_instances[agent_name]
        result = await agent.handle_action_request(action_request.json())
        
        return f"Step {step.id} execution started with {agent_name}: {result}"
    
    async def run_group_chat(self, user_input: str, plan_id: str = "", step_id: str = "") -> str:
        """Run the AgentGroupChat with a given input.
        
        Args:
            user_input: The user input to start the conversation
            plan_id: Optional plan ID for context
            step_id: Optional step ID for context
            
        Returns:
            Result of the group chat
        """
        # Ensure the group chat is initialized
        await self.initialize_group_chat()
        
        try:
            # Run the group chat with Semantic Kernel
            result = await self._agent_group_chat.invoke_async(user_input)
            
            # Process the result which could be a ChatHistory object or something else
            if hasattr(result, "messages") and result.messages:
                messages = result.messages
            elif hasattr(result, "value") and isinstance(result.value, list):
                messages = result.value
            else:
                # Fallback for other formats
                messages = []
                if isinstance(result, str):
                    # If it's just a string response
                    logging.debug(f"Group chat returned a string: {result[:100]}...")
                    await self._memory_store.add_item(
                        AgentMessage(
                            session_id=self._session_id,
                            user_id=self._user_id,
                            plan_id=plan_id,
                            content=result,
                            source="GroupChatManager",
                            step_id=step_id,
                        )
                    )
                    return result
            
            # Process the messages from the chat result
            final_response = None
            
            for msg in messages:
                # Skip the initial user message
                if hasattr(msg, "role") and msg.role == "user" and msg.content == user_input:
                    continue
                
                # Determine the source/agent name
                source = "assistant"
                if hasattr(msg, "name") and msg.name:
                    source = msg.name
                elif hasattr(msg, "role") and msg.role:
                    source = msg.role
                
                # Get the message content
                content = msg.content if hasattr(msg, "content") else str(msg)
                
                # Store the message in memory
                await self._memory_store.add_item(
                    AgentMessage(
                        session_id=self._session_id,
                        user_id=self._user_id,
                        plan_id=plan_id,
                        content=content,
                        source=source,
                        step_id=step_id,
                    )
                )
                
                # Keep track of the final response
                final_response = content
            
            # Return the final message from the chat
            if final_response:
                return final_response
            return "Group chat completed with no messages."
            
        except Exception as e:
            logging.exception(f"Error running group chat: {e}")
            return f"Error running group chat: {str(e)}"
    
    async def execute_next_step(self, kernel_arguments: KernelArguments) -> str:
        """Execute the next step in the plan.
        
        Args:
            kernel_arguments: KernelArguments that should contain session_id and plan_id
            
        Returns:
            Status message
        """
        # Extract arguments
        session_id = kernel_arguments.get("session_id", "")
        plan_id = kernel_arguments.get("plan_id", "")
        
        if not session_id or not plan_id:
            return "Missing session_id or plan_id in arguments"
        
        # Get all steps for the plan
        steps = await self._memory_store.get_steps_for_plan(plan_id, session_id)
        
        # Find the next step to execute (first approved or planned step)
        next_step = None
        for step in steps:
            if step.status == StepStatus.approved or step.status == StepStatus.planned:
                next_step = step
                break
        
        if not next_step:
            # All steps are completed, mark plan as completed
            plan = await self._memory_store.get_plan(plan_id)
            if plan:
                plan.overall_status = PlanStatus.completed
                await self._memory_store.update_plan(plan)
            return "All steps completed. Plan execution finished."
        
        return await self._execute_step(session_id, next_step)
    
    async def _generate_conversation_history(self, steps: List[Step], current_step_id: str, plan: Any) -> str:
        """Generate conversation history for context.
        
        Args:
            steps: List of all steps
            current_step_id: ID of the current step
            plan: The plan object
            
        Returns:
            Formatted conversation history
        """
        # Initialize the formatted string
        formatted_string = "<conversation_history>Here is the conversation history so far for the current plan. This information may or may not be relevant to the step you have been asked to execute."
        
        # Add plan summary if available
        if hasattr(plan, 'summary') and plan.summary:
            formatted_string += f"The user's task was:\n{plan.summary}\n\n"
        elif hasattr(plan, 'initial_goal') and plan.initial_goal:
            formatted_string += f"The user's task was:\n{plan.initial_goal}\n\n"
            
        formatted_string += "The conversation between the previous agents so far is below:\n"
        
        # Iterate over the steps until the current_step_id
        for i, step in enumerate(steps):
            if step.id == current_step_id:
                break
                
            if step.status == StepStatus.completed and hasattr(step, 'agent_reply') and step.agent_reply:
                formatted_string += f"Step {i}\n"
                formatted_string += f"Group chat manager: {step.action}\n"
                formatted_string += f"{step.agent}: {step.agent_reply}\n"
                
        formatted_string += "</conversation_history>"
        return formatted_string