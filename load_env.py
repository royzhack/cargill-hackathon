"""
Simple environment variable loader from .env file
"""

import os
from pathlib import Path


def load_env_file(env_file: str = ".env") -> dict:
    """
    Load environment variables from a .env file.
    
    Args:
        env_file: Path to .env file (default: .env in current directory)
    
    Returns:
        Dictionary of key-value pairs from .env file
    """
    env_path = Path(env_file)
    env_vars = {}
    
    if not env_path.exists():
        return env_vars
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                env_vars[key] = value
    
    return env_vars


def setup_env_from_file(env_file: str = ".env"):
    """
    Load .env file and set environment variables.
    
    This function reads the .env file and sets environment variables
    that can be accessed via os.getenv().
    
    Args:
        env_file: Path to .env file (default: .env in current directory)
    """
    env_vars = load_env_file(env_file)
    
    for key, value in env_vars.items():
        # Only set if not already in environment (env vars take precedence)
        if key not in os.environ:
            os.environ[key] = value


if __name__ == "__main__":
    # Test loading
    print("Loading .env file...")
    env_vars = load_env_file()
    
    if env_vars:
        print(f"✓ Loaded {len(env_vars)} variables from .env")
        for key in env_vars.keys():
            print(f"  - {key}")
    else:
        print("⚠ No variables found in .env file")
        print("  Make sure .env file exists and contains TEAM_API_KEY and SHARED_OPENAI_KEY")
