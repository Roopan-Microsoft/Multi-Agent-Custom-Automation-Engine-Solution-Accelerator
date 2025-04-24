# app_config.py
import os
import logging
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.cosmos.aio import CosmosClient
from azure.ai.projects.aio import AIProjectClient
from semantic_kernel.kernel import Kernel
from semantic_kernel.contents import ChatHistory
from semantic_kernel.agents.azure_ai.azure_ai_agent import AzureAIAgent
# Load environment variables from .env file
load_dotenv()

class AppConfig:
    """Application configuration class that loads settings from environment variables."""
    
    def __init__(self):
        """Initialize the application configuration with environment variables."""
        # Azure authentication settings
        self.AZURE_TENANT_ID = self._get_optional("AZURE_TENANT_ID")
        self.AZURE_CLIENT_ID = self._get_optional("AZURE_CLIENT_ID")
        self.AZURE_CLIENT_SECRET = self._get_optional("AZURE_CLIENT_SECRET")
        
        # CosmosDB settings
        self.COSMOSDB_ENDPOINT = self._get_optional("COSMOSDB_ENDPOINT")
        self.COSMOSDB_DATABASE = self._get_optional("COSMOSDB_DATABASE")
        self.COSMOSDB_CONTAINER = self._get_optional("COSMOSDB_CONTAINER")
        
        # Azure OpenAI settings
        self.AZURE_OPENAI_DEPLOYMENT_NAME = self._get_required("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        self.AZURE_OPENAI_API_VERSION = self._get_required("AZURE_OPENAI_API_VERSION", "2024-11-20")
        self.AZURE_OPENAI_ENDPOINT = self._get_required("AZURE_OPENAI_ENDPOINT")
        self.AZURE_OPENAI_SCOPES = [f"{self._get_optional('AZURE_OPENAI_SCOPE', 'https://cognitiveservices.azure.com/.default')}"]
        
        # Frontend settings
        self.FRONTEND_SITE_NAME = self._get_optional("FRONTEND_SITE_NAME", "http://127.0.0.1:3000")
        
        # Azure AI settings
        self.AZURE_AI_SUBSCRIPTION_ID = self._get_required("AZURE_AI_SUBSCRIPTION_ID")
        self.AZURE_AI_RESOURCE_GROUP = self._get_required("AZURE_AI_RESOURCE_GROUP")
        self.AZURE_AI_PROJECT_NAME = self._get_required("AZURE_AI_PROJECT_NAME")
        self.AZURE_AI_AGENT_PROJECT_CONNECTION_STRING = self._get_required("AZURE_AI_AGENT_PROJECT_CONNECTION_STRING")
        
        # Cached clients and resources
        self._azure_credentials = None
        self._cosmos_client = None
        self._cosmos_database = None
        self._ai_project_client = None
    
    def _get_required(self, name: str, default: Optional[str] = None) -> str:
        """Get a required configuration value from environment variables.
        
        Args:
            name: The name of the environment variable
            default: Optional default value if not found
            
        Returns:
            The value of the environment variable or default if provided
            
        Raises:
            ValueError: If the environment variable is not found and no default is provided
        """
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            logging.warning("Environment variable %s not found, using default value", name)
            return default
        raise ValueError(f"Environment variable {name} not found and no default provided")
    
    def _get_optional(self, name: str, default: str = "") -> str:
        """Get an optional configuration value from environment variables.
        
        Args:
            name: The name of the environment variable
            default: Default value if not found (default: "")
            
        Returns:
            The value of the environment variable or the default value
        """
        if name in os.environ:
            return os.environ[name]
        return default
    
    def _get_bool(self, name: str) -> bool:
        """Get a boolean configuration value from environment variables.
        
        Args:
            name: The name of the environment variable
            
        Returns:
            True if the environment variable exists and is set to 'true' or '1', False otherwise
        """
        return name in os.environ and os.environ[name].lower() in ["true", "1"]
    
    def get_azure_credentials(self):
        """Get Azure credentials using DefaultAzureCredential.
        
        Returns:
            DefaultAzureCredential instance for Azure authentication
        """
        # Cache the credentials object
        if self._azure_credentials is not None:
            return self._azure_credentials
        
        try:
            self._azure_credentials = DefaultAzureCredential()
            return self._azure_credentials
        except Exception as exc:
            logging.warning("Failed to create DefaultAzureCredential: %s", exc)
            return None
    
    def get_cosmos_database_client(self):
        """Get a Cosmos DB client for the configured database.
        
        Returns:
            A Cosmos DB database client
        """
        try:
            if self._cosmos_client is None:
                self._cosmos_client = CosmosClient(
                    self.COSMOSDB_ENDPOINT, credential=self.get_azure_credentials()
                )

            if self._cosmos_database is None:
                self._cosmos_database = self._cosmos_client.get_database_client(
                    self.COSMOSDB_DATABASE
                )

            return self._cosmos_database
        except Exception as exc:
            logging.error("Failed to create CosmosDB client: %s. CosmosDB is required for this application.", exc)
            raise
    
    def create_kernel(self):
        """Creates a new Semantic Kernel instance.
        
        Returns:
            A new Semantic Kernel instance
        """
        # Create a new kernel instance without manually configuring OpenAI services
        # The agents will be created using Azure AI Agent Project pattern instead
        kernel = Kernel()
        return kernel
    
    def get_ai_project_client(self):
        """Create and return an AIProjectClient for Azure AI Foundry using from_connection_string.
        
        Returns:
            An AIProjectClient instance
        """
        if self._ai_project_client is not None:
            return self._ai_project_client
            
        try:
            credential = self.get_azure_credentials()
            if credential is None:
                raise RuntimeError("Unable to acquire Azure credentials; ensure DefaultAzureCredential is configured")
                
            connection_string = self.AZURE_AI_AGENT_PROJECT_CONNECTION_STRING
            self._ai_project_client = AIProjectClient.from_connection_string(
                credential=credential,
                conn_str=connection_string
            )
            logging.info("Successfully created AIProjectClient using connection string")
            return self._ai_project_client
        except Exception as exc:
            logging.error("Failed to create AIProjectClient: %s", exc)
            raise
    
    async def create_azure_ai_agent(
        self,
        kernel: Kernel, 
        agent_name: str, 
        instructions: str, 
        agent_type: str = "assistant", 
        tools=None,
        tool_resources=None,
        response_format=None,
        temperature: float = 0.0
    ):
        """
        Creates a new Azure AI Agent with the specified name and instructions using AIProjectClient.
        
        Args:
            kernel: The Semantic Kernel instance
            agent_name: The name of the agent
            instructions: The system message / instructions for the agent
            agent_type: The type of agent (defaults to "assistant")
            tools: Optional tool definitions for the agent
            tool_resources: Optional tool resources required by the tools
            response_format: Optional response format to control structured output
            temperature: The temperature setting for the agent (defaults to 0.0)
            
        Returns:
            A new AzureAIAgent instance
        """
        try:
            # Get the AIProjectClient
            project_client = self.get_ai_project_client()
            
            # Tool handling: We need to distinguish between our SK functions and
            # the tool definitions needed by project_client.agents.create_agent
            tool_definitions = None
            kernel_functions = []
            
            # If tools are provided and they are SK KernelFunctions, we need to handle them differently
            # than if they are already tool definitions expected by AIProjectClient
            if tools:
                # Check if tools are SK KernelFunctions
                if all(hasattr(tool, 'name') and hasattr(tool, 'invoke') for tool in tools):
                    # Store the kernel functions to register with the agent later
                    kernel_functions = tools
                    # For now, we don't extract tool definitions from kernel functions
                    # This would require additional code to convert SK functions to AI Project tool definitions
                    logging.warning("Kernel functions provided as tools will be registered with the agent after creation")
                else:
                    # Assume these are already proper tool definitions for create_agent
                    tool_definitions = tools
            
            # Create the agent using the project client
            if response_format is not None:
                logging.info("Response format provided: %s", response_format)

                
            agent_definition = await project_client.agents.create_agent(
                model=self.AZURE_OPENAI_DEPLOYMENT_NAME,
                name=agent_name,
                instructions=instructions,
                tools=tool_definitions,  # Only pass tool_definitions, not kernel functions
                tool_resources=tool_resources,
                temperature=temperature,
                response_format=response_format
            )
            
            # Create the agent instance directly with project_client and definition
            agent_kwargs = {
                "client": project_client,
                "definition": agent_definition,
                "kernel": kernel
            }
            
   
            # For other agents, create using standard AzureAIAgent
            agent = AzureAIAgent(**agent_kwargs)
            
            # Register the kernel functions with the agent if any were provided
            if kernel_functions:
                for function in kernel_functions:
                    if hasattr(agent, 'add_function'):
                        agent.add_function(function)
                    
            return agent
        except Exception as exc:
            logging.error("Failed to create Azure AI Agent: %s", exc)
            raise


# Create a global instance of AppConfig
config = AppConfig()