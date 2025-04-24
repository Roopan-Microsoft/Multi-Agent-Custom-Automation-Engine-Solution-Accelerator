import logging
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Callable, Awaitable

import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.agents.azure_ai.azure_ai_agent import AzureAIAgent

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

from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import (
    ActionRequest,
    ActionResponse,
    AgentMessage,
    Step,
    StepStatus,
)
# Import the new AppConfig instance
from app_config import config
from event_utils import track_event_if_configured

# Default formatting instructions used across agents
DEFAULT_FORMATTING_INSTRUCTIONS = "Instructions: returning the output of this function call verbatim to the user in markdown. Then write AGENT SUMMARY: and then include a summary of what you did."

class BaseAgent(AzureAIAgent):
    """BaseAgent implemented using Semantic Kernel with Azure AI Agent support."""

    def __init__(
        self,
        agent_name: str,
        kernel: sk.Kernel,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: Optional[List[KernelFunction]] = None,
        system_message: Optional[str] = None,
        agent_type: Optional[str] = None,
        client=None,
        definition=None,
    ):
        """Initialize the base agent.
        
        Args:
            agent_name: The name of the agent
            kernel: The semantic kernel instance
            session_id: The session ID
            user_id: The user ID
            memory_store: The memory context for storing agent state
            tools: Optional list of tools for the agent
            system_message: Optional system message for the agent
            agent_type: Optional agent type string for automatic tool loading
            client: The client required by AzureAIAgent
            definition: The definition required by AzureAIAgent
        """
        # If agent_type is provided, load tools from config automatically
        if agent_type and not tools:
            tools = self.get_tools_from_config(kernel, agent_type)
            # If system_message isn't provided, try to get it from config
            if not system_message:
                config = self.load_tools_config(agent_type)
                system_message = config.get("system_message", self._default_system_message(agent_name))
        else:
            tools = tools or []
        system_message = system_message or self._default_system_message(agent_name)
        
        # Call AzureAIAgent constructor with required client and definition
        super().__init__(
            kernel=kernel,
            deployment_name=None,  # Set as needed
            endpoint=None,        # Set as needed
            api_version=None,     # Set as needed
            token=None,           # Set as needed
            agent_name=agent_name,
            system_prompt=system_message,
            client=client,
            definition=definition
        )
        
        # Store instance variables
        self._agent_name = agent_name
        self._kernel = kernel
        self._session_id = session_id
        self._user_id = user_id
        self._memory_store = memory_store
        self._tools = tools
        self._system_message = system_message
        self._chat_history = [{"role": "system", "content": self._system_message}]
        self._agent = None  # Will be initialized in async_init
        
        # Required properties for AgentGroupChat compatibility
        self.name = agent_name  # This is crucial for AgentGroupChat to identify agents
        
        
        # Register the handler functions
        self._register_functions()

    def _default_system_message(self, agent_name=None) -> str:
        name = agent_name or getattr(self, '_agent_name', 'Agent')
        return f"You are an AI assistant named {name}. Help the user by providing accurate and helpful information."

    async def async_init(self):
        """Asynchronously initialize the agent after construction.
        
        This method must be called after creating the agent to complete initialization.
        """
        # Create Azure AI Agent or fallback
        self._agent = await config.create_azure_ai_agent(
            kernel=self._kernel,
            agent_name=self._agent_name,
            instructions=self._system_message
        )
        # Tools are registered with the kernel via get_tools_from_config
        return self

    async def invoke_async(self, *args, **kwargs):
        """Invoke this agent asynchronously.
        
        This method is required for compatibility with AgentGroupChat.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            The agent's response
        """
        # Ensure agent is initialized
        if self._agent is None:
            await self.async_init()
            
        # Get the text input from args or kwargs
        text = None
        if args and isinstance(args[0], str):
            text = args[0]
        elif "text" in kwargs:
            text = kwargs["text"]
        elif "arguments" in kwargs and hasattr(kwargs["arguments"], "get"):
            text = kwargs["arguments"].get("text") or kwargs["arguments"].get("input")
        
        if not text:
            settings = kwargs.get("settings", {})
            if isinstance(settings, dict) and "input" in settings:
                text = settings["input"]
        
        # If text is still not found, create a default message
        if not text:
            text = "Hello, please assist with a task."
            
        # Use the text to invoke the agent
        try:
            logging.info(f"Invoking {self._agent_name} with text: {text[:100]}...")
            response = await self._agent.invoke(
                self._kernel, 
                text,
                settings=kwargs.get("settings", {})
            )
            return response
        except Exception as e:
            logging.error(f"Error invoking {self._agent_name}: {e}")
            return f"Error: {str(e)}"

    def _register_functions(self):
        """Register this agent's functions with the kernel."""
        # Use the kernel function decorator approach instead of from_native_method
        # which isn't available in SK 1.28.0
        function_name = "handle_action_request"
        
        # Define the function using the kernel function decorator
        @kernel_function(
            description="Handle an action request from another agent or the system",
            name=function_name
        )
        async def handle_action_request_wrapper(*args, **kwargs):
            # Forward to the instance method
            return await self.handle_action_request(*args, **kwargs)
            
        # Wrap the decorated function into a KernelFunction and register under this agent's plugin
        kernel_func = KernelFunction.from_method(handle_action_request_wrapper)
        # Use agent name as plugin for handler
        self._kernel.add_function(self._agent_name, kernel_func)
        
    # Required method for AgentGroupChat compatibility
    async def send_message_async(self, message_content: ChatMessageContent, chat_history: ChatHistory):
        """Send a message to the agent asynchronously, adding it to chat history.
        
        Args:
            message_content: The content of the message
            chat_history: The chat history
            
        Returns:
            None
        """
        # Convert message to format expected by the agent
        if hasattr(message_content, "role") and hasattr(message_content, "content"):
            self._chat_history.append({
                "role": message_content.role,
                "content": message_content.content
            })
        
        # If chat history is provided, update our internal history
        if chat_history and hasattr(chat_history, "messages"):
            # Update with the latest messages from chat history
            for msg in chat_history.messages[-5:]:  # Only use last 5 messages to avoid history getting too long
                if msg not in self._chat_history:
                    self._chat_history.append({
                        "role": msg.role, 
                        "content": msg.content
                    })
        
        # No need to return anything as we're just updating state
        return None

    async def handle_action_request(self, action_request_json: str) -> str:
        """Handle an action request from another agent or the system.
        
        Args:
            action_request_json: The action request as a JSON string
            
        Returns:
            A JSON string containing the action response
        """
        # Parse the action request
        action_request_dict = json.loads(action_request_json)
        action_request = ActionRequest(**action_request_dict)
        
        # Get the step from memory
        step: Step = await self._memory_store.get_step(
            action_request.step_id, action_request.session_id
        )
        
        if not step:
            # Create error response if step not found
            response = ActionResponse(
                step_id=action_request.step_id,
                status=StepStatus.failed,
                message="Step not found in memory.",
            )
            return response.json()
        
        # Add messages to chat history for context
        # This gives the agent visibility of the conversation history
        self._chat_history.extend([
            {"role": "assistant", "content": action_request.action},
            {"role": "user", "content": f"{step.human_feedback}. Now make the function call"}
        ])
        
        try:
            # Use the agent to process the action
            chat_history = self._chat_history.copy()
            
            # Call the agent to handle the action
            agent_response = await self._agent.invoke(self._kernel, f"{action_request.action}\n\nPlease perform this action")
            result = str(agent_response)
            
            # Store agent message in cosmos memory
            await self._memory_store.add_item(
                AgentMessage(
                    session_id=action_request.session_id,
                    user_id=self._user_id,
                    plan_id=action_request.plan_id,
                    content=f"{result}",
                    source=self._agent_name,
                    step_id=action_request.step_id,
                )
            )

            # Track telemetry
            track_event_if_configured(
                "Base agent - Added into the cosmos",
                {
                    "session_id": action_request.session_id,
                    "user_id": self._user_id,
                    "plan_id": action_request.plan_id,
                    "content": f"{result}",
                    "source": self._agent_name,
                    "step_id": action_request.step_id,
                },
            )

        except Exception as e:
            logging.exception(f"Error during agent execution: {e}")
            
            # Track error in telemetry
            track_event_if_configured(
                "Base agent - Error during agent execution, captured into the cosmos",
                {
                    "session_id": action_request.session_id,
                    "user_id": self._user_id,
                    "plan_id": action_request.plan_id,
                    "content": f"{e}",
                    "source": self._agent_name,
                    "step_id": action_request.step_id,
                },
            )
            
            # Return an error response
            response = ActionResponse(
                step_id=action_request.step_id,
                plan_id=action_request.plan_id,
                session_id=action_request.session_id,
                result=f"Error: {str(e)}",
                status=StepStatus.failed,
            )
            return response.json()
            
        logging.info(f"Task completed: {result}")

        # Update step status
        step.status = StepStatus.completed
        step.agent_reply = result
        await self._memory_store.update_step(step)

        # Track step completion in telemetry
        track_event_if_configured(
            "Base agent - Updated step and updated into the cosmos",
            {
                "status": StepStatus.completed,
                "session_id": action_request.session_id,
                "agent_reply": f"{result}",
                "user_id": self._user_id,
                "plan_id": action_request.plan_id,
                "content": f"{result}",
                "source": self._agent_name,
                "step_id": action_request.step_id,
            },
        )

        # Create and return action response
        response = ActionResponse(
            step_id=step.id,
            plan_id=step.plan_id,
            session_id=action_request.session_id,
            result=result,
            status=StepStatus.completed,
        )
        
        return response.json()

    async def invoke_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Invoke a specific tool by name with the provided arguments.
        
        Args:
            tool_name: The name of the tool to invoke
            arguments: A dictionary of arguments to pass to the tool
            
        Returns:
            The result of the tool invocation as a string
            
        Raises:
            ValueError: If the tool is not found
        """
        # Find the tool by name in the agent's tools list
        tool = next((t for t in self._tools if t.name == tool_name), None)
        
        if not tool:
            # Try looking up the tool in the kernel's plugins
            plugin_name = f"{self._agent_name.lower().replace('agent', '')}_plugin"
            try:
                tool = self._kernel.get_function(plugin_name, tool_name)
            except Exception:
                raise ValueError(f"Tool '{tool_name}' not found in agent tools or kernel plugins")
        
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
            
        try:
            # Create kernel arguments from the dictionary
            kernel_args = KernelArguments()
            for key, value in arguments.items():
                kernel_args[key] = value
                
            # Invoke the tool
            logging.info(f"Invoking tool '{tool_name}' with arguments: {arguments}")
            
            # Use invoke_with_args_dict directly instead of relying on KernelArguments
            if hasattr(tool, 'invoke_with_args_dict') and callable(tool.invoke_with_args_dict):
                result = await tool.invoke_with_args_dict(arguments)
            else:
                # Fall back to standard invoke method
                result = await tool.invoke(kernel_args)
            
            # Log telemetry if configured
            track_event_if_configured("AgentToolInvocation", {
                "agent_name": self._agent_name,
                "tool_name": tool_name,
                "session_id": self._session_id,
                "user_id": self._user_id
            })
            
            return str(result)
        except Exception as e:
            logging.error(f"Error invoking tool '{tool_name}': {str(e)}")
            raise

    @staticmethod
    def create_dynamic_function(name: str, response_template: str, formatting_instr: str = DEFAULT_FORMATTING_INSTRUCTIONS) -> Callable[..., Awaitable[str]]:
        """Create a dynamic function for agent tools based on the name and template.
        
        Args:
            name: The name of the function to create
            response_template: The template string to use for the response
            formatting_instr: Optional formatting instructions to append to the response
            
        Returns:
            A dynamic async function that can be registered with the semantic kernel
        """
        async def dynamic_function(**kwargs) -> str:
            try:
                # Format the template with the provided kwargs
                formatted_response = response_template.format(**kwargs)
                # Append formatting instructions if not already included in the template
                if formatting_instr and formatting_instr not in formatted_response:
                    formatted_response = f"{formatted_response}\n{formatting_instr}"
                return formatted_response
            except KeyError as e:
                return f"Error: Missing parameter {e} for {name}"
            except Exception as e:
                return f"Error processing {name}: {str(e)}"
        
        # Name the function properly for better debugging
        dynamic_function.__name__ = name
        
        # Create a wrapped kernel function that matches the expected signature
        @kernel_function(
            description=f"Dynamic function: {name}",
            name=name
        )
        async def kernel_wrapper(kernel_arguments: KernelArguments = None, **kwargs) -> str:
            # Combine all arguments into one dictionary
            all_args = {}
            if kernel_arguments:
                for key, value in kernel_arguments.items():
                    all_args[key] = value
            all_args.update(kwargs)
            return await dynamic_function(**all_args)
        
        return kernel_wrapper

    @staticmethod
    def load_tools_config(agent_type: str, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load tools configuration from a JSON file.
        
        Args:
            agent_type: The type of agent (e.g., "marketing", "hr")
            config_path: Optional explicit path to the configuration file
            
        Returns:
            A dictionary containing the configuration
        """
        if config_path is None:
            # Default path relative to the tools directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.dirname(current_dir)  # Just one level up to get to backend dir
            config_path = os.path.join(backend_dir, "tools", f"{agent_type}_tools.json")
        
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading {agent_type} tools configuration: {e}")
            # Return empty default configuration
            return {
                "agent_name": f"{agent_type.capitalize()}Agent",
                "system_message": "You are an AI assistant",
                "tools": []
            }

    @classmethod
    def get_tools_from_config(cls, kernel: sk.Kernel, agent_type: str, config_path: Optional[str] = None) -> List[KernelFunction]:
        """Get the list of tools for an agent from configuration.
        
        Args:
            kernel: The semantic kernel instance
            agent_type: The type of agent (e.g., "marketing", "hr")
            config_path: Optional explicit path to the configuration file
            
        Returns:
            A list of KernelFunction objects representing the tools
        """
        # Load configuration
        config = cls.load_tools_config(agent_type, config_path)
        
        # Convert the configured tools to kernel functions
        kernel_functions = []
        plugin_name = f"{agent_type}_plugin"
        
        # Early return if no tools defined - prevent empty iteration
        if not config.get("tools"):
            logging.info(f"No tools defined for agent type '{agent_type}'. Returning empty list.")
            return kernel_functions
        
        for tool in config.get("tools", []):
            try:
                function_name = tool["name"]
                description = tool.get("description", "")
                # Create a dynamic function using the JSON response_template
                response_template = tool.get("response_template") or tool.get("prompt_template") or ""
                
                # Generate a dynamic function using our improved approach
                dynamic_fn = cls.create_dynamic_function(function_name, response_template)
                
                # Create kernel function from the decorated function
                kernel_func = KernelFunction.from_method(dynamic_fn)
                
                # Add parameter metadata from JSON to the kernel function
                for param in tool.get("parameters", []):
                    param_name = param.get("name", "")
                    param_desc = param.get("description", "")
                    param_type = param.get("type", "string")
                    
                    # Set this parameter in the function's metadata
                    if param_name:
                        logging.debug(f"Adding parameter '{param_name}' to function '{function_name}'")
                
                # Register the function with the kernel
                kernel.add_function(plugin_name, kernel_func)
                kernel_functions.append(kernel_func)
                logging.debug(f"Successfully created dynamic tool '{function_name}' for {agent_type}")
            except Exception as e:
                logging.error(f"Failed to create tool '{tool.get('name', 'unknown')}': {str(e)}")
                
                
        return kernel_functions

    def save_state(self) -> Mapping[str, Any]:
        """Save the state of this agent."""
        return {"memory": self._memory_store.save_state()}

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Load the state of this agent."""
        self._memory_store.load_state(state["memory"])