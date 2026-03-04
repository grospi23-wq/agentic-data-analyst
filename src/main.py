import os
from dotenv import load_dotenv

load_dotenv()

def scan_api_keys():
    providers = {
        "Anthropic": ["ANTHROPIC_API_KEY"],
        "OpenAI": ["OPENAI_API_KEY"],
        "Gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "Groq": ["GROQ_API_KEY"]
    }
    
    active_keys = {}
    
    print("--- API Key Diagnostics ---")
    
    for provider, env_vars in providers.items():
        provider_found = False
        
        for env_var in env_vars:
            key_value = os.getenv(env_var)
            if key_value:
                masked = f"{key_value[:6]}...{key_value[-4:]}" if len(key_value) > 10 else "***"
                print(f"[OK] {provider:<10} -> Found in {env_var} ({masked})")
                active_keys[provider] = key_value
                provider_found = True
                break
        
        if not provider_found:
            print(f"[--] {provider:<10} -> Missing (Checked: {', '.join(env_vars)})")
            
    print("-" * 27)
    return active_keys

def main():
    available_providers = scan_api_keys()
    
    if not available_providers:
        print("System Halted: No valid API keys found.")
        return

    print(f"System ready. Total active providers: {len(available_providers)}")

if __name__ == "__main__":
    main()