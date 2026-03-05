"""
Test script for VoyageChatbot
Tests chatbot initialization and data loading without making API calls.
"""

import os
import json
from pathlib import Path
from voyage_chatbot import VoyageChatbot


def test_data_loading():
    """Test that optimization data files can be loaded."""
    print("=" * 70)
    print("TESTING DATA LOADING")
    print("=" * 70)
    
    portfolio_path = Path("processed/portfolio_summary.json")
    risk_path = Path("processed/portfolio_summary_risk_adjusted.json")
    
    # Check files exist
    assert portfolio_path.exists(), f"Portfolio data not found: {portfolio_path}"
    assert risk_path.exists(), f"Risk-adjusted data not found: {risk_path}"
    print("✓ Data files found")
    
    # Load and validate structure
    with open(portfolio_path) as f:
        portfolio = json.load(f)
    
    with open(risk_path) as f:
        risk = json.load(f)
    
    assert "assignments" in portfolio, "Portfolio data missing 'assignments'"
    assert "assignments" in risk, "Risk data missing 'assignments'"
    print(f"✓ Portfolio data: {len(portfolio['assignments'])} assignments")
    print(f"✓ Risk-adjusted data: {len(risk['assignments'])} assignments")
    
    # Validate assignment structure
    if portfolio['assignments']:
        assignment = portfolio['assignments'][0]
        required_fields = ['Vessel_Name', 'Cargo_ID', 'Leg_Profit', 'TCE_Leg']
        for field in required_fields:
            assert field in assignment, f"Assignment missing required field: {field}"
        print("✓ Assignment structure validated")
    
    return True


def test_chatbot_initialization():
    """Test chatbot initialization (without API calls)."""
    print("\n" + "=" * 70)
    print("TESTING CHATBOT INITIALIZATION")
    print("=" * 70)
    
    # Use dummy keys for testing (won't make actual API calls)
    try:
        chatbot = VoyageChatbot(
            team_api_key="test-team-key",
            shared_openai_key="test-shared-key",
            model="global.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        print("✓ Chatbot initialized successfully")
        print(f"✓ System prompt length: {len(chatbot.system_prompt)} characters")
        print(f"✓ Conversation history initialized: {len(chatbot.conversation_history)} messages")
        
        # Test data access
        portfolio = chatbot.get_portfolio_summary()
        risk = chatbot.get_risk_adjusted_summary()
        print(f"✓ Portfolio data accessible: {len(portfolio.get('assignments', []))} assignments")
        print(f"✓ Risk data accessible: {len(risk.get('assignments', []))} assignments")
        
        # Test conversation reset
        chatbot.reset_conversation()
        assert len(chatbot.conversation_history) == 1, "Reset should leave only system message"
        print("✓ Conversation reset works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_data_formatting():
    """Test that data formatting functions work correctly."""
    print("\n" + "=" * 70)
    print("TESTING DATA FORMATTING")
    print("=" * 70)
    
    chatbot = VoyageChatbot(
        team_api_key="test-team-key",
        shared_openai_key="test-shared-key"
    )
    
    # Test portfolio summary formatting
    portfolio_summary = chatbot._format_portfolio_summary()
    assert "Portfolio Overview" in portfolio_summary
    assert "Total Portfolio Profit" in portfolio_summary
    print("✓ Portfolio summary formatting works")
    
    # Test risk summary formatting
    risk_summary = chatbot._format_risk_summary()
    assert "Risk-Adjusted Portfolio" in risk_summary
    assert "Base Profit" in risk_summary
    print("✓ Risk summary formatting works")
    
    # Test data injection
    test_query = "What is the profit for PACIFIC GLORY?"
    enhanced = chatbot._inject_data_context(test_query)
    assert "PACIFIC GLORY" in enhanced
    assert "Assignment Details" in enhanced or "Profit" in enhanced
    print("✓ Data context injection works")
    
    return True


def test_model_validation():
    """Test that model validation works."""
    print("\n" + "=" * 70)
    print("TESTING MODEL VALIDATION")
    print("=" * 70)
    
    # Test valid model
    try:
        chatbot = VoyageChatbot(
            team_api_key="test-key",
            shared_openai_key="test-key",
            model="global.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        print("✓ Valid model accepted")
    except Exception as e:
        print(f"✗ Valid model rejected: {e}")
        return False
    
    # Test invalid model
    try:
        chatbot = VoyageChatbot(
            team_api_key="test-key",
            shared_openai_key="test-key",
            model="invalid-model-name"
        )
        print("✗ Invalid model accepted (should have failed)")
        return False
    except ValueError:
        print("✓ Invalid model correctly rejected")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("VOYAGE CHATBOT TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Chatbot Initialization", test_chatbot_initialization),
        ("Data Formatting", test_data_formatting),
        ("Model Validation", test_model_validation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Chatbot is ready for use.")
        print("\nNote: To use the chatbot with actual API calls, set:")
        print("  - TEAM_API_KEY environment variable")
        print("  - SHARED_OPENAI_KEY environment variable")
    else:
        print("\n✗ Some tests failed. Please review the errors above.")


if __name__ == "__main__":
    main()

