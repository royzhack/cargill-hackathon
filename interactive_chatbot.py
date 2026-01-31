"""
Interactive Chatbot Demo
========================
Simple command-line interface for the VoyageChatbot.
"""

import os
import sys
from pathlib import Path
from voyage_chatbot import VoyageChatbot


def get_api_keys():
    """
    Get API keys from config file, .env file, or environment variables.
    Priority: chatbot_config.txt > .env file > environment variables
    """
    # Try chatbot_config.txt first
    try:
        from load_env import load_env_file
        config_vars = load_env_file("chatbot_config.txt")
        if config_vars.get('TEAM_API_KEY') and config_vars.get('SHARED_OPENAI_KEY') and \
           config_vars.get('TEAM_API_KEY') != "your-team-api-key-here":
            return (
                config_vars.get('TEAM_API_KEY'),
                config_vars.get('SHARED_OPENAI_KEY'),
                config_vars.get('CHATBOT_MODEL')
            )
    except:
        pass
    
    # Try .env file
    try:
        from load_env import load_env_file
        env_vars = load_env_file(".env")
        if env_vars.get('TEAM_API_KEY') and env_vars.get('SHARED_OPENAI_KEY'):
            return (
                env_vars.get('TEAM_API_KEY'),
                env_vars.get('SHARED_OPENAI_KEY'),
                env_vars.get('CHATBOT_MODEL')
            )
    except:
        pass
    
    # Fall back to environment variables
    return (
        os.getenv("TEAM_API_KEY"),
        os.getenv("SHARED_OPENAI_KEY"),
        os.getenv("CHATBOT_MODEL")
    )


def main():
    """Run interactive chatbot session."""
    print("=" * 70)
    print("VOYAGE OPTIMIZATION CHATBOT - INTERACTIVE MODE")
    print("=" * 70)
    print()
    
    # Get API keys
    team_api_key, shared_openai_key, model = get_api_keys()
    
    if not team_api_key or not shared_openai_key or team_api_key == "your-team-api-key-here":
        print("⚠️  Error: API keys not found")
        print()
        print("Please update chatbot_config.txt with your API keys:")
        print("  1. Open chatbot_config.txt")
        print("  2. Replace 'your-team-api-key-here' with your actual TEAM_API_KEY")
        print("  3. Replace 'your-shared-openai-key-here' with your actual SHARED_OPENAI_KEY")
        print()
        print("Alternatively, you can:")
        print("  - Create a .env file with TEAM_API_KEY and SHARED_OPENAI_KEY")
        print("  - Set environment variables: export TEAM_API_KEY='key'")
        print()
        sys.exit(1)
    
    # Initialize chatbot
    print("Initializing chatbot...")
    try:
        chatbot = VoyageChatbot(
            team_api_key=team_api_key,
            shared_openai_key=shared_openai_key,
            model=model or "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        print("✓ Chatbot initialized successfully")
        print(f"✓ Loaded {len(chatbot.portfolio_data.get('assignments', []))} assignments")
        print()
    except Exception as e:
        print(f"✗ Error initializing chatbot: {e}")
        sys.exit(1)
    
    # Interactive loop
    print("=" * 70)
    print("CHATBOT READY")
    print("=" * 70)
    print()
    print("Type your questions about voyage optimization.")
    print("Commands:")
    print("  'reset' - Reset conversation history")
    print("  'exit' or 'quit' - Exit the chatbot")
    print()
    print("-" * 70)
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye!")
                break
            
            if user_input.lower() == 'reset':
                chatbot.reset_conversation()
                print("\n✓ Conversation history reset")
                continue
            
            # Get response
            print("\nAssistant: ", end="", flush=True)
            response = chatbot.chat(user_input)
            print(response)
            print("-" * 70)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            print("Please try again or type 'exit' to quit.")


if __name__ == "__main__":
    main()

