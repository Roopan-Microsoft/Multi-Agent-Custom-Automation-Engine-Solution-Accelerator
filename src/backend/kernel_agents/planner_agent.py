import logging
import uuid
import json
import re
import datetime
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.functions.kernel_arguments import KernelArguments

from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import (
    AgentMessage,
    InputTask,
    Plan,
    PlannerResponsePlan,
    Step,
    StepStatus,
    PlanStatus,
    HumanFeedbackStatus,
)
from event_utils import track_event_if_configured
from app_config import config


class PlannerAgent(BaseAgent):
    """Planner agent implementation using Semantic Kernel.
    
    This agent creates and manages plans based on user tasks, breaking them down into steps
    that can be executed by specialized agents to achieve the user's goal.
    """

    def __init__(
        self,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: Optional[List[KernelFunction]] = None,
        system_message: Optional[str] = None,
        agent_name: str = "PlannerAgent",
        config_path: Optional[str] = None,
        available_agents: List[str] = None,
        agent_tools_list: List[str] = None,
        agent_instances: Optional[Dict[str, BaseAgent]] = None,
        client=None,
        definition=None,
    ) -> None:
        """Initialize the Planner Agent.
        
        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: Optional list of tools for this agent
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "PlannerAgent")
            config_path: Optional path to the configuration file
            available_agents: List of available agent names for creating steps
            agent_tools_list: List of available tools across all agents
            agent_instances: Dictionary of agent instances available to the planner
            client: Optional client instance (passed to BaseAgent)
            definition: Optional definition instance (passed to BaseAgent)
        """
        # Default system message if not provided
        if not system_message:
            system_message = "You are a Planner agent responsible for creating and managing plans. You analyze tasks, break them down into steps, and assign them to the appropriate specialized agents."
        
        # Initialize the base agent
        super().__init__(
            agent_name=agent_name,
            kernel=kernel,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=tools,
            system_message=system_message,
            agent_type="planner",  # Use planner_tools.json if available
            client=client,
            definition=definition
        )
        
        # Store additional planner-specific attributes
        self._available_agents = available_agents or ["HumanAgent", "HrAgent", "MarketingAgent", 
                                                     "ProductAgent", "ProcurementAgent", 
                                                     "TechSupportAgent", "GenericAgent"]
        self._agent_tools_list = agent_tools_list or []
        self._agent_instances = agent_instances or {}
        
        # Create the Azure AI Agent for planning operations
        # This will be initialized in async_init
        self._azure_ai_agent = None

    def _get_response_format_schema(self) -> dict:
        """
        Returns a JSON schema that defines the expected structure of the response.
        This ensures responses from the agent will match the required format exactly.
        """
        return {
            "type": "object",
            "properties": {
                "initial_goal": {
                    "type": "string",
                    "description": "The primary goal extracted from the user's input task"
                },
                "steps": {
                    "type": "array",
                    "description": "List of steps required to complete the task",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "A clear instruction for the agent including the function name to use"
                            },
                            "agent": {
                                "type": "string",
                                "description": "The name of the agent responsible for this step"
                            }
                        },
                        "required": ["action", "agent"]
                    }
                },
                "summary_plan_and_steps": {
                    "type": "string",
                    "description": "A concise summary of the overall plan and its steps in less than 50 words"
                },
                "human_clarification_request": {
                    "type": ["string", "null"],
                    "description": "Optional request for additional information needed from the user"
                }
            },
            "required": ["initial_goal", "steps", "summary_plan_and_steps"]
        }
        
    async def async_init(self) -> None:
        """Asynchronously initialize the PlannerAgent.
        
        Creates the Azure AI Agent for planning operations.
        
        Returns:
            None
        """
        try:
            logging.info("Initializing PlannerAgent from async init azure AI Agent")
            # Create the Azure AI Agent using AppConfig
            self._azure_ai_agent = await config.create_azure_ai_agent(
                kernel=self._kernel,
                agent_name="PlannerAgent",
                instructions=self._generate_instruction(""),
                temperature=0.0
            )
            logging.info("Successfully created Azure AI Agent for PlannerAgent")
            return True
        except Exception as e:
            logging.error(f"Failed to create Azure AI Agent for PlannerAgent: {e}")
            raise
        
    async def handle_input_task(self, input_task: InputTask) -> str:
        """Handle the initial input task from the user.
        
        Args:
            kernel_arguments: Contains the input_task_json string
            
        Returns:
            Status message
        """
        # Parse the input task
        logging.info("Handling input task")


        plan, steps = await self._create_structured_plan(input_task)

        logging.info(f"Plan created: {plan}")
        logging.info(f"Steps created: {steps}")


        
        if steps:
            # Add a message about the created plan
            await self._memory_store.add_item(
                AgentMessage(
                    session_id=input_task.session_id,
                    user_id=self._user_id,
                    plan_id=plan.id,
                    content=f"Generated a plan with {len(steps)} steps. Click the blue check box beside each step to complete it, click the x to remove this step.",
                    source="PlannerAgent",
                    step_id="",
                )
            )
            
            track_event_if_configured(
                f"Planner - Generated a plan with {len(steps)} steps and added plan into the cosmos",
                {
                    "session_id": input_task.session_id,
                    "user_id": self._user_id,
                    "plan_id": plan.id,
                    "content": f"Generated a plan with {len(steps)} steps. Click the blue check box beside each step to complete it, click the x to remove this step.",
                    "source": "PlannerAgent",
                },
            )
            
            # If human clarification is needed, add a message requesting it
            if hasattr(plan, 'human_clarification_request') and plan.human_clarification_request:
                await self._memory_store.add_item(
                    AgentMessage(
                        session_id=input_task.session_id,
                        user_id=self._user_id,
                        plan_id=plan.id,
                        content=f"I require additional information before we can proceed: {plan.human_clarification_request}",
                        source="PlannerAgent",
                        step_id="",
                    )
                )
                
                track_event_if_configured(
                    "Planner - Additional information requested and added into the cosmos",
                    {
                        "session_id": input_task.session_id,
                        "user_id": self._user_id,
                        "plan_id": plan.id,
                        "content": f"I require additional information before we can proceed: {plan.human_clarification_request}",
                        "source": "PlannerAgent",
                    },
                )
        
        return f"Plan '{plan.id}' created successfully with {len(steps)} steps"
        
    async def handle_plan_clarification(self, kernel_arguments: KernelArguments) -> str:
        """Handle human clarification for a plan.
        
        Args:
            kernel_arguments: Contains session_id and human_clarification
            
        Returns:
            Status message
        """
        session_id = kernel_arguments["session_id"]
        human_clarification = kernel_arguments["human_clarification"]
        
        # Retrieve and update the plan
        plan = await self._memory_store.get_plan_by_session(session_id)
        if not plan:
            return f"No plan found for session {session_id}"
            
        plan.human_clarification_response = human_clarification
        await self._memory_store.update_plan(plan)
        
        # Add a record of the clarification
        await self._memory_store.add_item(
            AgentMessage(
                session_id=session_id,
                user_id=self._user_id,
                plan_id="",
                content=f"{human_clarification}",
                source="HumanAgent",
                step_id="",
            )
        )
        
        track_event_if_configured(
            "Planner - Store HumanAgent clarification and added into the cosmos",
            {
                "session_id": session_id,
                "user_id": self._user_id,
                "content": f"{human_clarification}",
                "source": "HumanAgent",
            },
        )
        
        # Add a confirmation message
        await self._memory_store.add_item(
            AgentMessage(
                session_id=session_id,
                user_id=self._user_id,
                plan_id="",
                content="Thanks. The plan has been updated.",
                source="PlannerAgent",
                step_id="",
            )
        )
        
        track_event_if_configured(
            "Planner - Updated with HumanClarification and added into the cosmos",
            {
                "session_id": session_id,
                "user_id": self._user_id,
                "content": "Thanks. The plan has been updated.",
                "source": "PlannerAgent",
            },
        )
        
        return "Plan updated with human clarification"
        
    async def _create_structured_plan(self, input_task: InputTask) -> Tuple[Plan, List[Step]]:
        """Create a structured plan with steps based on the input task.
        
        Args:
            input_task: The input task from the user
            
        Returns:
            Tuple containing the created plan and list of steps
        """
        try:
            # Generate the instruction for the LLM
            logging.info("Generating instruction for the LLM")
            logging.debug(f"Input: {input_task}")
            logging.debug(f"Available agents: {self._available_agents}")

            instruction = self._generate_instruction(input_task.description)
            
            logging.info(f"Generated instruction: {instruction}")   
            # Log the input task for debugging
            logging.info(f"Creating plan for task: '{input_task.description}'")
            logging.info(f"Using available agents: {self._available_agents}")
            
            # Use the Azure AI Agent instead of direct function invocation
            if self._azure_ai_agent is None:
                # Initialize the agent if it's not already done
                await self.async_init()
                
            if self._azure_ai_agent is None:
                raise RuntimeError("Failed to initialize Azure AI Agent for planning")
                
            # Log detailed information about the instruction being sent
            logging.info(f"Invoking PlannerAgent with instruction length: {len(instruction)}")
            
            # Create kernel arguments - make sure we explicitly emphasize the task
            kernel_args = KernelArguments()
            kernel_args["input"] = f"TASK: {input_task.description}\n\n{instruction}"
            
            logging.debug(f"Kernel arguments: {kernel_args}")

            # Get the schema for our expected response format
            response_format_schema = self._get_response_format_schema()
            
            # Call invoke with proper keyword arguments and JSON response schema
            response_content = ""
            
            # Ensure we're using the right pattern for Azure AI agents with semantic kernel
            # Properly handle async generation
            async_generator = self._azure_ai_agent.invoke(
                arguments=kernel_args,
                settings={
                    "temperature": 0.0,  # Keep temperature low for consistent planning
                    "max_tokens": 10096,  # Ensure we have enough tokens for the full plan
                    "response_format": {
                        "type": "json_object",
                        "schema": response_format_schema
                    }
                }
            )
            
            # Collect the response from the async generator
            async for chunk in async_generator:
                if chunk is not None:
                    response_content += str(chunk)
            
            logging.info(f"Response content length: {len(response_content)}")
            logging.debug(f"Response content: {response_content[:500]}...")
            
            # Check if response is empty or whitespace
            if not response_content or response_content.isspace():
                raise ValueError("Received empty response from Azure AI Agent")
            
            # Parse the JSON response directly to PlannerResponsePlan
            parsed_result = None
            
            # Try various parsing approaches in sequence
            try:
                # 1. First attempt: Try to parse the raw response directly
                try:
                    parsed_result = PlannerResponsePlan.parse_raw(response_content)
                    logging.info("Successfully parsed response with direct parsing")
                except Exception as parse_error:
                    logging.warning(f"Failed direct parse: {parse_error}")
                    
                    # 2. Try to extract JSON from markdown code blocks
                    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_content, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1)
                        logging.info(f"Found JSON in code block, attempting to parse")
                        try:
                            parsed_result = PlannerResponsePlan.parse_raw(json_content)
                            logging.info("Successfully parsed JSON from code block")
                        except Exception as code_block_error:
                            logging.warning(f"Failed to parse JSON in code block: {code_block_error}")
                            # Try parsing as dict first, then convert to model
                            try:
                                json_dict = json.loads(json_content)
                                parsed_result = PlannerResponsePlan.parse_obj(json_dict)
                                logging.info("Successfully parsed JSON dict from code block")
                            except Exception as dict_error:
                                logging.warning(f"Failed to parse JSON dict from code block: {dict_error}")
                    
                    # 3. Look for patterns like { ... } that might contain JSON
                    if parsed_result is None:
                        json_pattern = r'\{.*?"initial_goal".*?"steps".*?\}'
                        alt_match = re.search(json_pattern, response_content, re.DOTALL)
                        if alt_match:
                            potential_json = alt_match.group(0)
                            logging.info(f"Found potential JSON pattern in text, attempting to parse")
                            try:
                                json_dict = json.loads(potential_json)
                                parsed_result = PlannerResponsePlan.parse_obj(json_dict)
                                logging.info("Successfully parsed JSON using regex pattern extraction")
                            except Exception as pattern_error:
                                logging.warning(f"Failed to parse JSON pattern: {pattern_error}")
                
                if parsed_result is None:
                    # If all parsing attempts fail, create a fallback plan from the text content
                    logging.warning("All JSON parsing attempts failed, creating fallback plan from text")
                    return await self._create_fallback_plan_from_text(input_task, response_content)
                    
            except Exception as parsing_exception:
                logging.exception(f"Error during parsing attempts: {parsing_exception}")
                return await self._create_fallback_plan_from_text(input_task, response_content)
            
            # At this point, we have a valid parsed_result
            
            # Extract plan details
            initial_goal = parsed_result.initial_goal
            steps_data = parsed_result.steps
            summary = parsed_result.summary_plan_and_steps
            human_clarification_request = parsed_result.human_clarification_request
            
            # Create the Plan instance
            plan = Plan(
                id=str(uuid.uuid4()),
                session_id=input_task.session_id,
                user_id=self._user_id,
                initial_goal=initial_goal,
                overall_status=PlanStatus.in_progress,
                summary=summary,
                human_clarification_request=human_clarification_request
            )
            
            # Store the plan
            await self._memory_store.add_plan(plan)
            
            # Create steps from the parsed data
            steps = []
            for step_data in steps_data:
                action = step_data.action
                agent_name = step_data.agent
                
                # Validate agent name
                if agent_name not in self._available_agents:
                    logging.warning(f"Invalid agent name: {agent_name}, defaulting to GenericAgent")
                    agent_name = "GenericAgent"
                
                # Create the step
                step = Step(
                    id=str(uuid.uuid4()),
                    plan_id=plan.id,
                    session_id=input_task.session_id,
                    user_id=self._user_id,
                    action=action,
                    agent=agent_name,
                    status=StepStatus.planned,
                    human_approval_status=HumanFeedbackStatus.requested
                )
                
                # Store the step
                await self._memory_store.add_step(step)
                steps.append(step)
                
                try:
                    track_event_if_configured(
                        "Planner - Added planned individual step into the cosmos",
                        {
                            "plan_id": plan.id,
                            "action": action,
                            "agent": agent_name,
                            "status": StepStatus.planned,
                            "session_id": input_task.session_id,
                            "user_id": self._user_id,
                            "human_approval_status": HumanFeedbackStatus.requested,
                        },
                    )
                except Exception as event_error:
                    # Don't let event tracking errors break the main flow
                    logging.warning(f"Error in event tracking: {event_error}")
            
            return plan, steps
                
        except Exception as e:
            logging.exception(f"Error creating structured plan: {e}")
            
            # Create a fallback dummy plan when parsing fails
            logging.info("Creating fallback dummy plan due to parsing error")
            
            import datetime
            
            # Create a dummy plan with the original task description
            dummy_plan = Plan(
                id=str(uuid.uuid4()),
                session_id=input_task.session_id,
                user_id=self._user_id,
                initial_goal=input_task.description,
                overall_status=PlanStatus.in_progress,
                summary=f"Plan created for: {input_task.description}",
                human_clarification_request=None,
                timestamp=datetime.datetime.utcnow().isoformat()
            )
            
            # Store the dummy plan
            await self._memory_store.add_plan(dummy_plan)
            
            # Create a dummy step for analyzing the task
            dummy_step = Step(
                id=str(uuid.uuid4()),
                plan_id=dummy_plan.id,
                session_id=input_task.session_id,
                user_id=self._user_id,
                action="Analyze the task: " + input_task.description,
                agent="GenericAgent",
                status=StepStatus.planned,
                human_approval_status=HumanFeedbackStatus.requested,
                timestamp=datetime.datetime.utcnow().isoformat()
            )
            
            # Store the dummy step
            await self._memory_store.add_step(dummy_step)
            
            # Add a second step to request human clarification
            clarification_step = Step(
                id=str(uuid.uuid4()),
                plan_id=dummy_plan.id,
                session_id=input_task.session_id,
                user_id=self._user_id,
                action=f"Provide more details about: {input_task.description}",
                agent="HumanAgent",
                status=StepStatus.planned,
                human_approval_status=HumanFeedbackStatus.requested,
                timestamp=datetime.datetime.utcnow().isoformat()
            )
            
            # Store the clarification step
            await self._memory_store.add_step(clarification_step)
            
            # Log the event
            try:
                track_event_if_configured(
                    "Planner - Created fallback dummy plan due to parsing error",
                    {
                        "session_id": input_task.session_id,
                        "user_id": self._user_id,
                        "error": str(e),
                        "description": input_task.description,
                        "source": "PlannerAgent",
                    }
                )
            except Exception as event_error:
                logging.warning(f"Error in event tracking during fallback: {event_error}")
            
            return dummy_plan, [dummy_step, clarification_step]
            
    async def _create_fallback_plan_from_text(self, input_task: InputTask, text_content: str) -> Tuple[Plan, List[Step]]:
        """Create a plan from unstructured text when JSON parsing fails.
        
        Args:
            input_task: The input task
            text_content: The text content from the LLM
            
        Returns:
            Tuple containing the created plan and list of steps
        """
        logging.info("Creating fallback plan from text content")
        
        # Extract goal from the text (first line or use input task description)
        goal_match = re.search(r"(?:Goal|Initial Goal|Plan):\s*(.+?)(?:\n|$)", text_content)
        goal = goal_match.group(1).strip() if goal_match else input_task.description
        
        # Create the plan
        plan = Plan(
            id=str(uuid.uuid4()),
            session_id=input_task.session_id,
            user_id=self._user_id,
            initial_goal=goal,
            overall_status=PlanStatus.in_progress,
            summary=f"Plan created from {input_task.description}"
        )
        
        # Store the plan
        await self._memory_store.add_plan(plan)
        
        # Parse steps using regex
        step_pattern = re.compile(r'(?:Step|)\s*(\d+)[:.]\s*\*?\*?(?:Agent|):\s*\*?([^:*\n]+)\*?[:\s]*(.+?)(?=(?:Step|)\s*\d+[:.]\s*|$)', re.DOTALL)
        matches = step_pattern.findall(text_content)
        
        if not matches:
            # Fallback to simpler pattern
            step_pattern = re.compile(r'(\d+)[.:\)]\s*([^:]*?):\s*(.*?)(?=\d+[.:\)]|$)', re.DOTALL)
            matches = step_pattern.findall(text_content)
        
        # If still no matches, look for bullet points or numbered lists
        if not matches:
            step_pattern = re.compile(r'[•\-*]\s*([^:]*?):\s*(.*?)(?=[•\-*]|$)', re.DOTALL)
            bullet_matches = step_pattern.findall(text_content)
            if bullet_matches:
                # Convert bullet matches to our expected format (number, agent, action)
                matches = []
                for i, (agent_text, action) in enumerate(bullet_matches, 1):
                    matches.append((str(i), agent_text.strip(), action.strip()))
        
        steps = []
        # If we found no steps at all, create at least one generic step
        if not matches:
            generic_step = Step(
                id=str(uuid.uuid4()),
                plan_id=plan.id,
                session_id=input_task.session_id,
                user_id=self._user_id,
                action=f"Process the request: {input_task.description}",
                agent="GenericAgent",
                status=StepStatus.planned,
                human_approval_status=HumanFeedbackStatus.requested
            )
            await self._memory_store.add_step(generic_step)
            steps.append(generic_step)
        else:
            for match in matches:
                number = match[0].strip()
                agent_text = match[1].strip()
                action = match[2].strip()
                
                # Clean up agent name
                agent = re.sub(r'\s+', '', agent_text)
                if not agent or agent not in self._available_agents:
                    agent = "GenericAgent"  # Default to GenericAgent if not recognized
                
                # Create and store the step
                step = Step(
                    id=str(uuid.uuid4()),
                    plan_id=plan.id,
                    session_id=input_task.session_id,
                    user_id=self._user_id,
                    action=action,
                    agent=agent,
                    status=StepStatus.planned,
                    human_approval_status=HumanFeedbackStatus.requested
                )
                
                await self._memory_store.add_step(step)
                steps.append(step)
                
        return plan, steps
    
    def _generate_instruction(self, objective: str) -> str:
        """Generate instruction for the LLM to create a plan.
        
        Args:
            objective: The user's objective
            
        Returns:
            Instruction string for the LLM
        """
        # Create a list of available agents
        agents_str = ", ".join(self._available_agents)
        
        # Create list of available tools in JSON-like format
        tools_list = []
        
        # Check if we have agent instances to extract tools from
        if hasattr(self, '_agent_instances') and self._agent_instances:
            # Process each agent to get their tools
            for agent_name, agent in self._agent_instances.items():
                if hasattr(agent, '_tools') and agent._tools:
                    # Add each tool from this agent
                    for tool in agent._tools:
                        if hasattr(tool, 'name') and hasattr(tool, 'description'):
                            # Extract function parameters/arguments
                            args_dict = {}
                            if hasattr(tool, 'parameters'):
                                # Check if we have kernel_arguments that need to be processed
                                has_kernel_args = any(param.name == 'kernel_arguments' for param in tool.parameters)
                                has_kwargs = any(param.name == 'kwargs' for param in tool.parameters)
                                
                                # Process regular parameters first
                                for param in tool.parameters:
                                    # Skip kernel_arguments and kwargs as we'll handle them specially
                                    if param.name in ['kernel_arguments', 'kwargs']:
                                        continue
                                    
                                    param_type = "string"  # Default type
                                    if hasattr(param, 'type'):
                                        param_type = param.type
                                    
                                    args_dict[param.name] = {
                                        'description': param.description if param.description else param.name,
                                        'title': param.name.replace('_', ' ').title(),
                                        'type': param_type
                                    }
                                
                                # If we have a kernel_arguments parameter, introspect it to extract its values
                                # This is a special case handling for kernel_arguments to include its fields in the arguments
                                if has_kernel_args:
                                    # Check if we have kernel_parameter_descriptions
                                    if hasattr(tool, 'kernel_parameter_descriptions'):
                                        # Extract parameter descriptions from the kernel
                                        for key, description in tool.kernel_parameter_descriptions.items():
                                            if key not in args_dict:  # Only add if not already added
                                                args_dict[key] = {
                                                    'description': description if description else key,
                                                    'title': key.replace('_', ' ').title(),
                                                    'type': 'string'  # Default to string type
                                                }
                                    # Fall back to function's description if no specific descriptions 
                                    elif hasattr(tool, 'description') and not args_dict:
                                        # Add a generic parameter with the function's description
                                        args_dict['input'] = {
                                            'description': f"Input for {tool.name}: {tool.description}",
                                            'title': 'Input',
                                            'type': 'string'
                                        }
                            
                            # If after all processing, arguments are still empty, add a dummy input parameter
                            if not args_dict:
                                args_dict['input'] = {
                                    'description': f"Input for {tool.name}",
                                    'title': 'Input',
                                    'type': 'string'
                                }
                            
                            # Create tool entry
                            tool_entry = {
                                'agent': agent_name,
                                'function': tool.name,
                                'description': tool.description,
                                'arguments': str(args_dict)
                            }
                            
                            tools_list.append(tool_entry)
            
            logging.debug(f"Generated {len(tools_list)} tools from agent instances")
        
        # If we couldn't extract tools from agent instances, create a simplified format
        if not tools_list:
            logging.warning("No tool details extracted from agent instances, creating simplified format")
            if self._agent_tools_list:
                # Create dummy entries from the existing tool list strings
                for tool_str in self._agent_tools_list:
                    if ":" in tool_str:
                        parts = tool_str.split(":")
                        if len(parts) >= 2:
                            agent_part = parts[0].strip()
                            function_part = parts[1].strip()
                            
                            # Extract agent name if format is "Agent: AgentName"
                            agent_name = agent_part.replace("Agent", "").strip()
                            if not agent_name:
                                agent_name = "GenericAgent"
                                
                            tools_list.append({
                                'agent': agent_name,
                                'function': function_part,
                                'description': f"Function {function_part} from {agent_name}",
                                'arguments': "{}"
                            })
        
        # Convert the tools list to a string representation
        tools_str = str(tools_list)
        
        # Build the instruction, avoiding backslashes in f-string expressions  

        instruction_template = f"""
You are the Planner, an AI orchestrator that manages a group of AI agents to accomplish tasks.

        For the given objective, come up with a simple step-by-step plan.
        This plan should involve individual tasks that, if executed correctly, will yield the correct answer. Do not add any superfluous steps.
        The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

        These actions are passed to the specific agent. Make sure the action contains all the information required for the agent to execute the task.
        
        Your objective is:
        {objective}

        The agents you have access to are:
        {agents_str}

        These agents have access to the following functions:
        {tools_str}

        IMPORTANT AGENT SELECTION GUIDANCE:
        - HrAgent: ALWAYS use for ALL employee-related tasks like onboarding, hiring, benefits, payroll, training, employee records, ID cards, mentoring, background checks, etc.
        - MarketingAgent: Use for marketing campaigns, branding, market research, content creation, social media, etc.
        - ProcurementAgent: Use for purchasing, vendor management, supply chain, asset management, etc.
        - ProductAgent: Use for product development, roadmaps, features, product feedback, etc.
        - TechSupportAgent: Use for technical issues, software/hardware setup, troubleshooting, IT support, etc.
        - GenericAgent: Use only for general knowledge tasks that don't fit other categories
        - HumanAgent: Use only when human input is absolutely required and no other agent can handle the task

        The first step of your plan should be to ask the user for any additional information required to progress the rest of steps planned.

        Only use the functions provided as part of your plan. If the task is not possible with the agents and tools provided, create a step with the agent of type Exception and mark the overall status as completed.

        Do not add superfluous steps - only take the most direct path to the solution, with the minimum number of steps. Only do the minimum necessary to complete the goal.

        If there is a single function call that can directly solve the task, only generate a plan with a single step. For example, if someone asks to be granted access to a database, generate a plan with only one step involving the grant_database_access function, with no additional steps.

        You must prioritise using the provided functions to accomplish each step. First evaluate each and every function the agents have access too. Only if you cannot find a function needed to complete the task, and you have reviewed each and every function, and determined why each are not suitable, there are two options you can take when generating the plan.
        First evaluate whether the step could be handled by a typical large language model, without any specialised functions. For example, tasks such as "add 32 to 54", or "convert this SQL code to a python script", or "write a 200 word story about a fictional product strategy".

        If a general Large Language Model CAN handle the step/required action, add a step to the plan with the action you believe would be needed, and add "EXCEPTION: No suitable function found. A generic LLM model is being used for this step." to the end of the action. Assign these steps to the GenericAgent. For example, if the task is to convert the following SQL into python code (SELECT * FROM employees;), and there is no function to convert SQL to python, write a step with the action "convert the following SQL into python code (SELECT * FROM employees;) EXCEPTION: No suitable function found. A generic LLM model is being used for this step." and assign it to the GenericAgent.

        Alternatively, if a general Large Language Model CAN NOT handle the step/required action, add a step to the plan with the action you believe would be needed, and add "EXCEPTION: Human support required to do this step, no suitable function found." to the end of the action. Assign these steps to the HumanAgent. For example, if the task is to find the best way to get from A to B, and there is no function to calculate the best route, write a step with the action "Calculate the best route from A to B. EXCEPTION: Human support required, no suitable function found." and assign it to the HumanAgent.

        Limit the plan to 6 steps or less.

        Choose from {agents_str} ONLY for planning your steps.

        """
        return instruction_template