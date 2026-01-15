"""
ReAct Agent with Real LLM Integration

This example shows how to use the ReAct agent with Anthropic's Claude API.
You'll need to install: pip install anthropic
And set your API key: export ANTHROPIC_API_KEY='your-key-here'
"""

from react_agent import ReActAgent, calculator, search


def create_claude_llm(model: str = "claude-sonnet-4-20250514", max_tokens: int = 1000):
    """
    Create an LLM callable that uses Claude API.
    
    Args:
        model: Claude model to use
        max_tokens: Maximum tokens in response
        
    Returns:
        Callable that takes prompt string and returns response string
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("Please install anthropic: pip install anthropic")
    
    import os
    # breakpoint()
    client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    
    def llm_call(prompt: str) -> str:
        """Call Claude API with the given prompt."""
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    
    return llm_call


def create_openai_llm(model: str = "gpt-4", max_tokens: int = 1000):
    """
    Create an LLM callable that uses OpenAI API.
    
    Args:
        model: OpenAI model to use
        max_tokens: Maximum tokens in response
        
    Returns:
        Callable that takes prompt string and returns response string
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")
    
    client = OpenAI()
    
    def llm_call(prompt: str) -> str:
        """Call OpenAI API with the given prompt."""
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    
    return llm_call


# Example usage
if __name__ == "__main__":
    
    # Choose your LLM backend
    # Uncomment one of the following:
    
    # Option 1: Anthropic Claude
    llm = create_claude_llm()
    
    # Option 2: OpenAI
    # llm = create_openai_llm()
    
    # For demo without API keys, use mock:
    # from react_agent import mock_llm
    # llm = mock_llm
    
    # Define tools
    tools = {
        "calculator": calculator,
        "search": search,
    }
    
    # Create and run agent
    agent = ReActAgent(llm_call=llm, tools=tools)
    
    question = "What is 123 * 456?"
    result = agent.run(question, verbose=True)
    
    print(f"\n\nFinal Result: {result}")