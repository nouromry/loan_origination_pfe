# src/agents/base_agent.py

import os
import json
import yaml
import re
from typing import List, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage


class BaseAgent:
    """
    Parent class for all agents in AXE Finance.
    
    HYBRID LLM STRATEGY:
    - Agents touching PII → Ollama (local, data never leaves the server)
    - Agents with no PII  → Groq (cloud, fast 70B models)
    
    Provider routing is driven entirely by config/settings.yaml.
    Switching an agent from local to cloud = changing one line in YAML.
    """

    # Cache for settings and prompts (loaded once per process)
    _settings_cache: Dict[str, Any] = None
    _prompts_cache: Dict[str, str] = None

    def __init__(self, model_name: str = None, provider: str = None):
        """
        Initialize the agent.
        
        Args:
            model_name: Override model name (if None, reads from settings.yaml)
            provider: Override provider 'ollama' or 'groq' (if None, reads from settings.yaml)
        """
        settings = self._load_settings()
        
        # Resolve model config from settings.yaml
        agent_key = self._get_agent_key()
        model_config = settings.get("models", {}).get(agent_key, {})
        
        # Handle both old flat format ("llama-3.1-8b-instant") and new dict format
        if isinstance(model_config, str):
            # Legacy flat format — default to groq
            resolved_provider = provider or "groq"
            resolved_model = model_name or model_config
        else:
            # New dict format with provider/model
            resolved_provider = provider or model_config.get("provider", "groq")
            resolved_model = model_name or model_config.get("model", "llama-3.1-8b-instant")

        # Create the LLM based on provider
        self.provider = resolved_provider
        self.llm = self._create_llm(resolved_provider, resolved_model)
        self.tools = self.get_tools()
        self.system_prompt = self._load_prompt()

        # Bind tools to LLM if any exist
        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm

    def _get_agent_key(self) -> str:
        """Convert prompt key to settings key (e.g., 'triage_agent' -> 'triage')."""
        return self.get_prompt_key().replace("_agent", "").replace("_node", "")

    @staticmethod
    def _create_llm(provider: str, model: str):
        """
        Factory method: create the right LLM client based on provider.
        
        This is the ONLY place in the entire codebase where LLM clients
        are instantiated. Switching providers = changing this one method.
        """
        if provider == "ollama":
            from langchain_ollama import ChatOllama
            
            settings = BaseAgent._load_settings()
            base_url = settings.get("ollama", {}).get("base_url", "http://localhost:11434")
            
            return ChatOllama(
                model=model,
                base_url=base_url,
                temperature=0.1,
            )

        elif provider == "groq":
            from langchain_groq import ChatGroq
            
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY not set in .env but agent is configured to use Groq. "
                    "Either set the key or switch this agent to 'ollama' in settings.yaml."
                )
            
            return ChatGroq(
                model=model,
                api_key=api_key,
                temperature=0.1,
            )

        else:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Supported: 'ollama' (local) or 'groq' (cloud)."
            )

    @classmethod
    def _load_settings(cls) -> Dict[str, Any]:
        """Load settings.yaml (cached at class level)."""
        if cls._settings_cache is None:
            settings_path = os.path.join(
                os.path.dirname(__file__), '../../config/settings.yaml'
            )
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    cls._settings_cache = yaml.safe_load(f)
            except FileNotFoundError:
                cls._settings_cache = {}
        return cls._settings_cache

    @classmethod
    def _load_prompts(cls) -> Dict[str, str]:
        """Load all prompts from agent_prompts.yaml (cached at class level)."""
        if cls._prompts_cache is None:
            prompt_path = os.path.join(
                os.path.dirname(__file__), '../prompts/agent_prompts.yaml'
            )
            with open(prompt_path, 'r', encoding='utf-8') as f:
                cls._prompts_cache = yaml.safe_load(f)
        return cls._prompts_cache

    def _load_prompt(self) -> str:
        """Load the specific prompt for this agent from the YAML file."""
        prompts = self._load_prompts()
        prompt_key = self.get_prompt_key()

        if prompt_key not in prompts:
            raise ValueError(f"Prompt key '{prompt_key}' not found in agent_prompts.yaml")
        return prompts[prompt_key]

    def get_tools(self) -> List[BaseTool]:
        """Override in child classes to assign specific tools."""
        return []

    def get_prompt_key(self) -> str:
        """Override in child classes to define the YAML prompt key."""
        raise NotImplementedError("Subclasses must implement get_prompt_key()")

    def run(self, goal: str, **context) -> Dict[str, Any]:
        """
        Execute the agent's reasoning loop.
        """
        # Format the system prompt with context variables
        formatted_prompt = self.system_prompt
        for key, value in context.items():
            formatted_prompt = formatted_prompt.replace(f"{{{key}}}", str(value))

        # Build messages
        messages = [
            SystemMessage(content=formatted_prompt),
            HumanMessage(content=f"GOAL: {goal}\n\nCONTEXT: {json.dumps(context, default=str)}"),
        ]

        # Execute
        if self.tools:
            response = self._run_with_tools(messages)
        else:
            response = self.llm.invoke(messages)

        return self._extract_result(response.content)

    def _run_with_tools(self, messages: list, max_iterations: int = 5):
        """Simple ReAct loop: call LLM -> execute tools -> repeat."""
        from langchain_core.messages import ToolMessage

        current_messages = list(messages)

        for _ in range(max_iterations):
            response = self.llm_with_tools.invoke(current_messages)
            current_messages.append(response)

            if not response.tool_calls:
                return response

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                tool_fn = None
                for t in self.tools:
                    if t.name == tool_name:
                        tool_fn = t
                        break

                if tool_fn is None:
                    result = f"Error: Tool '{tool_name}' not found"
                else:
                    try:
                        result = tool_fn.invoke(tool_args)
                    except Exception as e:
                        result = f"Error executing {tool_name}: {str(e)}"

                current_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )

        return current_messages[-1] if current_messages else response

    def _extract_result(self, content: str) -> Dict[str, Any]:
        """Extract structured JSON from the agent's response text."""
        if not content:
            return {"raw_response": "", "error": "Empty response from agent"}

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        brace_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return {"raw_response": content}
