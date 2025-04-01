import pytest
import os
from unittest.mock import patch, AsyncMock, MagicMock
from pyseoanalyzer.llm_analyst import (
    LLMSEOEnhancer,
    EntityAnalysis, # Import Pydantic models for mocking if needed
    CredibilityAnalysis,
    ConversationAnalysis,
    PlatformPresence,
    SEORecommendations
)
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatLiteLLM
from langchain_ollama import ChatOllama # Corrected import
from langchain.chains import LLMChain
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
        # Add other keys that might be expected by _format_output or enhance_seo_analysis
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


def test_init(monkeypatch):
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
    assert enhancer_anthropic_default.llm.model == "claude-3-5-haiku-latest"
    monkeypatch.delenv("ANTHROPIC_API_KEY") # Clean up

    # --- Test LiteLLM Path ---
    monkeypatch.setenv("LLM_PROVIDER", "litellm")
    monkeypatch.setenv("LITELLM_HOST", "http://fake-host:8000")
    monkeypatch.setenv("OLLAMA_MODEL", "test-ollama-model") # Base name
    monkeypatch.setenv("LITELLM_API_KEY", "fake-litellm-key") # Test with API key
    enhancer_litellm = LLMSEOEnhancer()
    assert isinstance(enhancer_litellm.llm, ChatLiteLLM)
    assert enhancer_litellm.llm.model == "ollama/test-ollama-model" # Check prefix is added
    assert enhancer_litellm.llm.openai_api_base == "http://fake-host:8000" # Check correct base
    assert enhancer_litellm.llm.api_key == "fake-litellm-key" # Check standard key param
    assert enhancer_litellm.llm.temperature == 0

    # Test LiteLLM without optional API key
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    enhancer_litellm_no_key = LLMSEOEnhancer()
    assert isinstance(enhancer_litellm_no_key.llm, ChatLiteLLM)
    assert enhancer_litellm_no_key.llm.model == "ollama/test-ollama-model"
    assert enhancer_litellm_no_key.llm.api_key is None # Check key is None
    monkeypatch.delenv("LITELLM_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    # --- Test Direct Ollama Path ---
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "test-ollama-direct")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama-host:11434") # Test custom base URL
    enhancer_ollama = LLMSEOEnhancer()
    assert isinstance(enhancer_ollama.llm, ChatOllama)
    assert enhancer_ollama.llm.model == "test-ollama-direct"
    assert enhancer_ollama.llm.base_url == "http://ollama-host:11434"
    assert enhancer_ollama.llm.temperature == 0

    # Test Ollama default base URL
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    enhancer_ollama_default_url = LLMSEOEnhancer()
    assert isinstance(enhancer_ollama_default_url.llm, ChatOllama)
    assert enhancer_ollama_default_url.llm.base_url == "http://localhost:11434" # Check default
    monkeypatch.delenv("OLLAMA_MODEL", raising=False) # Clean up

    # --- Test Error Path ---
    monkeypatch.setenv("LLM_PROVIDER", "unsupported")
    with pytest.raises(ValueError, match="LLM configuration error"):
        LLMSEOEnhancer()

    monkeypatch.setenv("LLM_PROVIDER", "litellm") # Set provider
    monkeypatch.delenv("LITELLM_HOST", raising=False) # Remove host
    monkeypatch.setenv("OLLAMA_MODEL", "some-model") # Add model back
    with pytest.raises(ValueError, match="LLM configuration error"):
        LLMSEOEnhancer() # Should fail without host

    monkeypatch.setenv("LITELLM_HOST", "some-host") # Add host back
    monkeypatch.delenv("OLLAMA_MODEL", raising=False) # Remove model
    with pytest.raises(ValueError, match="LLM configuration error"):
        LLMSEOEnhancer() # Should fail without model

    monkeypatch.setenv("LLM_PROVIDER", "ollama") # Set provider
    monkeypatch.delenv("OLLAMA_MODEL", raising=False) # Remove model
    with pytest.raises(ValueError, match="LLM configuration error"):
        LLMSEOEnhancer() # Should fail without model

    # Clean up all test env vars
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LITELLM_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
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
    enhancer.recommendations_chain.ainvoke = AsyncMock(return_value=MockSEORecommendations()) # Corrected target

    # Call the method under test
    result = await enhancer.enhance_seo_analysis(seo_data)

    # Assertions
    assert "summary" in result # Check if summary formatting worked (even with mock data)
    assert "detailed_analysis" in result
    assert "quick_wins" in result
    assert "strategic_recommendations" in result

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
    enhancer.recommendations_chain.ainvoke.assert_awaited_once() # Corrected target


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

    # Recommendations chain shouldn't be called if a prior chain failed
    enhancer.recommendations_chain.ainvoke = AsyncMock() # Corrected target

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
    assert "error" in detailed["recommendations"] # Recommendations should show error

    # Check chains were called appropriately
    mock_gather.assert_awaited_once()
    enhancer.recommendations_chain.ainvoke.assert_not_awaited() # Corrected target: Recommendations should not be invoked


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
    enhancer.recommendations_chain.ainvoke = AsyncMock(side_effect=ValueError("Simulated recommendation error")) # Corrected target

    result = await enhancer.enhance_seo_analysis(seo_data)

    # Assertions
    detailed = result["detailed_analysis"]
    assert "entity_analysis" in detailed
    assert "credibility_analysis" in detailed
    assert "conversation_analysis" in detailed
    assert "cross_platform_presence" in detailed
    assert "recommendations" in detailed
    assert "error" in detailed["recommendations"] # Check error was recorded
    assert "Failed to generate recommendations" in detailed["recommendations"]["error"]

    # Check chains were called
    mock_gather.assert_awaited_once()
    enhancer.recommendations_chain.ainvoke.assert_awaited_once() # Corrected target: Recommendations was invoked but failed
