import pytest
import os
from typing import List, Dict, Union # Import necessary types
from unittest.mock import patch, AsyncMock, MagicMock
from pyseoanalyzer.llm_analyst import (
    LLMSEOEnhancer,
    EntityAnalysis,
    CredibilityAnalysis,
    ConversationAnalysis,
    PlatformPresence,
    SEORecommendations
)
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatLiteLLM # Keep import for isinstance check if needed, though tests removed
from langchain_ollama import ChatOllama
from langchain.chains import LLMChain # Keep import if needed elsewhere, though not used in patched tests
from langchain.prompts import PromptTemplate
import json


@pytest.fixture
def seo_data():
    """ Basic SEO data fixture """
    return {
        "title": "Test Title",
        "description": "Test Description",
        "keywords": [{"word": "test", "count": 5}, {"word": "seo", "count": 3}], # Use dict format
        "content": "This is a test content for analysis.",
        "url": "http://example.com",
        "author": "Test Author",
        "hostname": "example.com",
        "sitename": "Example Site",
        "date": "2025-04-01",
        "word_count": 6,
        "bigrams": {},
        "trigrams": {},
        "warnings": [],
        "content_hash": "testhash",
        "headings": {},
        "additional_info": {}
    }

# Mock Pydantic models for return values of mocked chains
class MockEntityAnalysis(EntityAnalysis):
    entity_assessment: str = "mock entity"
    knowledge_panel_readiness: int = 1
    key_improvements: List[str] = ["e1"]

class MockCredibilityAnalysis(CredibilityAnalysis):
    credibility_assessment: str = "mock cred"
    neeat_scores: Dict[str, int] = {"N": 1, "E1": 1, "E2": 1, "A": 1, "T1": 1, "T2": 1} # Ensure 6 keys for division
    trust_signals: List[str] = ["c1"]

class MockConversationAnalysis(ConversationAnalysis):
    conversation_readiness: str = "mock convo"
    query_patterns: List[str] = ["q1"]
    engagement_score: int = 2
    gaps: List[str] = ["g1"]

class MockPlatformPresence(PlatformPresence):
    platform_coverage: Dict[str, str] = {"p1": "good"}
    visibility_scores: Dict[str, int] = {"v1": 3}
    optimization_opportunities: List[str] = ["o1"]

class MockSEORecommendations(SEORecommendations):
    strategic_recommendations: List[str] = ["s1"]
    quick_wins: List[str] = ["qw1"]
    long_term_strategy: List[str] = ["lts1"]
    priority_matrix: Dict[str, List[str]] = {"pm1": ["p1"]} # Use List[str]


@patch('pyseoanalyzer.llm_analyst.load_dotenv', return_value=None) # Patch load_dotenv to prevent override
def test_init(mock_load_dotenv, monkeypatch): # Add mock_load_dotenv to signature
    """ Tests the __init__ method for different providers """
    # --- Test Anthropic Path ---
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    monkeypatch.setenv("LLM_MODEL", "test-anthropic-model")
    enhancer_anthropic = LLMSEOEnhancer()
    assert isinstance(enhancer_anthropic.llm, ChatAnthropic)
    assert enhancer_anthropic.llm.model == "test-anthropic-model"
    assert enhancer_anthropic.llm.temperature == 0
    monkeypatch.delenv("LLM_MODEL", raising=False) # Clean up model env var

    # Test Anthropic default model
    enhancer_anthropic_default = LLMSEOEnhancer()
    assert isinstance(enhancer_anthropic_default.llm, ChatAnthropic)
    # Update expected default model based on llm_analyst.py code
    assert enhancer_anthropic_default.llm.model == "claude-3-sonnet-20240229"
    monkeypatch.delenv("ANTHROPIC_API_KEY") # Clean up

    # --- Test Direct Ollama Path ---
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "test-ollama-direct") # Use LLM_MODEL as checked in code
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama-host:11434") # Test custom base URL
    enhancer_ollama = LLMSEOEnhancer()
    assert isinstance(enhancer_ollama.llm, ChatOllama)
    assert enhancer_ollama.llm.model == "test-ollama-direct"
    assert enhancer_ollama.llm.base_url == "http://ollama-host:11434"
    assert enhancer_ollama.llm.temperature == 0

    # Test Ollama default base URL
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    # Need to set LLM_MODEL even when testing default base URL
    monkeypatch.setenv("LLM_MODEL", "test-ollama-default-url")
    enhancer_ollama_default_url = LLMSEOEnhancer()
    assert isinstance(enhancer_ollama_default_url.llm, ChatOllama)
    assert enhancer_ollama_default_url.llm.base_url == "http://localhost:11434" # Check default
    monkeypatch.delenv("OLLAMA_MODEL", raising=False) # Clean up

    # --- Test Error Path ---
    monkeypatch.setenv("LLM_PROVIDER", "unsupported")
    # Expect specific error message from the updated code
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER: 'unsupported'"):
        LLMSEOEnhancer()

    # Test missing Ollama model
    monkeypatch.setenv("LLM_PROVIDER", "ollama") # Set provider
    monkeypatch.delenv("LLM_MODEL", raising=False) # Remove model
    with pytest.raises(ValueError, match="LLM_MODEL is required for Ollama provider"):
        LLMSEOEnhancer() # Should fail without model

    # Clean up all test env vars
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)


@pytest.mark.asyncio
@patch('pyseoanalyzer.llm_analyst.LLMSEOEnhancer._setup_chains', return_value=None) # Prevent real chain setup
@patch('pyseoanalyzer.llm_analyst.asyncio.gather', new_callable=AsyncMock) # Mock asyncio.gather
async def test_enhance_seo_analysis_mocked(mock_gather, mock_setup, seo_data, monkeypatch):
    """Tests the enhance_seo_analysis method with mocked LLM calls."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic") # Use any valid provider for init
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    enhancer = LLMSEOEnhancer()

    # Manually mock the chains on the instance since _setup_chains is mocked
    enhancer.entity_chain = MagicMock()
    enhancer.credibility_chain = MagicMock()
    enhancer.conversation_chain = MagicMock()
    enhancer.platform_chain = MagicMock()
    enhancer.recommendations_chain = MagicMock() # Keep this mock too

    # Define mock return values for the gather call (results of individual chains)
    mock_gather.return_value = [
        MockEntityAnalysis(),
        MockCredibilityAnalysis(),
        MockConversationAnalysis(),
        MockPlatformPresence()
    ]

    # Define mock return value for the recommendations chain
    enhancer.recommendations_chain.ainvoke = AsyncMock(return_value=MockSEORecommendations())

    # Call the method under test
    result = await enhancer.enhance_seo_analysis(seo_data)

    # Assertions
    assert "summary" in result # Check if summary formatting worked (even with mock data)
    assert "detailed_analysis" in result
    assert "quick_wins" in result
    assert "strategic_recommendations" in result
    assert "errors" in result and not result["errors"] # Check errors list is present and empty

    # Check detailed analysis structure
    detailed = result["detailed_analysis"]
    assert "entity_analysis" in detailed
    assert detailed["entity_analysis"]["entity_assessment"] == "mock entity"
    assert "credibility_analysis" in detailed
    assert detailed["credibility_analysis"]["credibility_assessment"] == "mock cred"
    assert "conversation_analysis" in detailed
    assert detailed["conversation_analysis"]["conversation_readiness"] == "mock convo"
    assert "cross_platform_presence" in detailed
    assert detailed["cross_platform_presence"]["platform_coverage"] == {"p1": "good"}
    assert "recommendations" in detailed # Check recommendations are nested here now
    assert detailed["recommendations"]["quick_wins"] == ["qw1"]

    # Check top-level recommendations
    assert result["quick_wins"] == ["qw1"]
    assert result["strategic_recommendations"] == ["s1"]

    # Check that gather was called
    mock_gather.assert_awaited_once()
    # Check that recommendations_chain.ainvoke was called
    enhancer.recommendations_chain.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
@patch('pyseoanalyzer.llm_analyst.LLMSEOEnhancer._setup_chains', return_value=None)
@patch('pyseoanalyzer.llm_analyst.asyncio.gather', new_callable=AsyncMock)
async def test_enhance_seo_analysis_gather_error(mock_gather, mock_setup, seo_data, monkeypatch):
    """Tests error handling when one analysis chain fails."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic") # Use any valid provider for init
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    enhancer = LLMSEOEnhancer()

    # Manually mock the chains on the instance since _setup_chains is mocked
    enhancer.entity_chain = MagicMock()
    enhancer.credibility_chain = MagicMock()
    enhancer.conversation_chain = MagicMock()
    enhancer.platform_chain = MagicMock()
    enhancer.recommendations_chain = MagicMock() # Keep this mock too

    # Simulate an error in one of the gathered tasks (e.g., credibility)
    mock_gather.return_value = [
        MockEntityAnalysis(),
        ValueError("Simulated credibility error"), # Error for credibility
        MockConversationAnalysis(),
        MockPlatformPresence()
    ]

    # Recommendations chain should still be invoked if other chains succeeded
    enhancer.recommendations_chain.ainvoke = AsyncMock(return_value=MockSEORecommendations())

    result = await enhancer.enhance_seo_analysis(seo_data)

    # Assertions
    detailed = result["detailed_analysis"]
    assert "entity_analysis" in detailed
    assert detailed["entity_analysis"]["entity_assessment"] == "mock entity"
    assert "credibility_analysis" in detailed
    assert "error" in detailed["credibility_analysis"] # Check error was recorded
    assert "Simulated credibility error" in detailed["credibility_analysis"]["error"]
    assert "conversation_analysis" in detailed
    assert "cross_platform_presence" in detailed
    assert "recommendations" in detailed
    assert "error" not in detailed["recommendations"] # Recommendations should succeed here

    # Check overall errors list
    assert "errors" in result
    assert len(result["errors"]) == 1
    assert "Error in chain 'credibility_analysis': Simulated credibility error" in result["errors"][0]

    # Check chains were called appropriately
    mock_gather.assert_awaited_once()
    # Recommendations *should* be invoked if other chains succeeded
    enhancer.recommendations_chain.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
@patch('pyseoanalyzer.llm_analyst.LLMSEOEnhancer._setup_chains', return_value=None)
@patch('pyseoanalyzer.llm_analyst.asyncio.gather', new_callable=AsyncMock)
async def test_enhance_seo_analysis_reco_error(mock_gather, mock_setup, seo_data, monkeypatch):
    """Tests error handling when the recommendations chain fails."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic") # Use any valid provider for init
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    enhancer = LLMSEOEnhancer()

    # Manually mock the chains on the instance since _setup_chains is mocked
    enhancer.entity_chain = MagicMock()
    enhancer.credibility_chain = MagicMock()
    enhancer.conversation_chain = MagicMock()
    enhancer.platform_chain = MagicMock()
    enhancer.recommendations_chain = MagicMock() # Keep this mock too

    # Simulate success in gathered tasks
    mock_gather.return_value = [
        MockEntityAnalysis(),
        MockCredibilityAnalysis(),
        MockConversationAnalysis(),
        MockPlatformPresence()
    ]

    # Simulate error in recommendations chain
    enhancer.recommendations_chain.ainvoke = AsyncMock(side_effect=ValueError("Simulated recommendation error"))

    result = await enhancer.enhance_seo_analysis(seo_data)

    # Assertions
    detailed = result["detailed_analysis"]
    assert "entity_analysis" in detailed
    assert "credibility_analysis" in detailed
    assert "conversation_analysis" in detailed
    assert "cross_platform_presence" in detailed
    assert "recommendations" in detailed
    assert "error" in detailed["recommendations"] # Check error was recorded
    # Check the error message format applied in llm_analyst.py
    assert "Failed to generate recommendations: Simulated recommendation error" in detailed["recommendations"]["error"]

    # Check overall errors list
    assert "errors" in result
    assert len(result["errors"]) == 1
    assert "Error generating recommendations: Simulated recommendation error" in result["errors"][0]

    # Check chains were called
    mock_gather.assert_awaited_once()
    enhancer.recommendations_chain.ainvoke.assert_awaited_once() # Recommendations was invoked but failed


@pytest.mark.asyncio
@patch('pyseoanalyzer.llm_analyst.load_dotenv', return_value=None) # Patch load_dotenv
@patch('langchain_anthropic.ChatAnthropic.ainvoke', new_callable=AsyncMock) # Use class patch
async def test_analyze_markdown_success(mock_llm_ainvoke, mock_load_dotenv, monkeypatch): # Add mock args
    """Tests the analyze_markdown method successfully returns recommendations."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic") # Use Anthropic for this test
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

    # Mock the LLM response (raw string expected by parser) - MUST match MockSEORecommendations structure
    mock_llm_output_str = json.dumps(MockSEORecommendations().model_dump()) # Use json.dumps for correct formatting
    # Return the raw string directly, as the parser expects string input from the LLM step
    mock_llm_ainvoke.return_value = mock_llm_output_str

    enhancer = LLMSEOEnhancer() # Initialize after setting env vars

    # The real parser will be used on the mocked LLM output string
    sample_markdown = "# Test Markdown\nSome content here."

    # Call the method under test
    result = await enhancer.analyze_markdown(sample_markdown)

    # Assertions
    assert isinstance(result, dict) # Should return a dict
    # Check specific fields instead of exact dict match for less brittleness
    expected_result = MockSEORecommendations().model_dump()
    assert result.get("quick_wins") == expected_result.get("quick_wins")
    assert result.get("strategic_recommendations") == expected_result.get("strategic_recommendations")

    # Verify the mocked llm's ainvoke was called
    mock_llm_ainvoke.assert_awaited_once()


@pytest.mark.asyncio
@patch('pyseoanalyzer.llm_analyst.load_dotenv', return_value=None) # Patch load_dotenv
@patch('langchain_anthropic.ChatAnthropic.ainvoke', new_callable=AsyncMock) # Use class patch
async def test_analyze_markdown_error(mock_llm_ainvoke, mock_load_dotenv, monkeypatch): # Add mock args
    """Tests error handling within the analyze_markdown method."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic") # Use Anthropic for this test
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

    # Mock the llm's ainvoke method (via class patch) to raise an error
    mock_llm_ainvoke.side_effect = Exception("Simulated LLM error")

    enhancer = LLMSEOEnhancer() # Initialize after setting env vars

    sample_markdown = "# Error Test Markdown"

    # Call the method and expect it to handle the error gracefully
    result = await enhancer.analyze_markdown(sample_markdown)

    # Assertions for error handling
    assert isinstance(result, dict)
    assert "error" in result
    # Check for the exact error message format returned by the except block
    assert result["error"] == "LLM analysis failed: Simulated LLM error"

    # Verify the mocked llm's ainvoke was called
    mock_llm_ainvoke.assert_awaited_once()
