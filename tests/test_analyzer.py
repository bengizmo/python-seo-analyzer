import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from pyseoanalyzer.analyzer import analyze


# Mock LLM results for testing
MOCK_LLM_RESULTS = {
    "strategic_recommendations": ["Strategy 1"],
    "quick_wins": ["Quick Win 1"],
    "long_term_strategy": ["Long Term 1"],
    "priority_matrix": {"High Impact / Low Effort": ["Action A"]},
    "errors": [] # Ensure errors list exists
}

# Mock Page data for standard analysis tests
MOCK_PAGE_DICT = {
    "url": "http://example.com",
    "title": "Example Title",
    "description": "Example Description",
    "word_count": 100,
    "load_time": 0.5,
    "headings": {"h1": ["Main Heading"]},
    "llm_analysis": None, # Standard analysis might not have LLM data initially
}

@pytest.fixture
def mock_website():
    """Fixture to mock the Website class and its methods."""
    with patch('pyseoanalyzer.analyzer.Website', autospec=True) as MockWebsite:
        mock_instance = MockWebsite.return_value
        # Make crawl an AsyncMock as it's now async
        mock_instance.crawl = AsyncMock(return_value=None)
        mock_instance.crawled_pages = [MagicMock(spec=['as_dict'])]
        mock_instance.crawled_pages[0].as_dict.return_value = MOCK_PAGE_DICT
        mock_instance.content_hashes = {}
        mock_instance.wordcount = {'example': 10, 'test': 5}
        mock_instance.bigrams = {('example', 'test'): 6}
        mock_instance.trigrams = {('example', 'test', 'run'): 5}
        mock_instance.errors = []
        mock_instance.llm_analysis = None # Simulate no site-level LLM analysis by default
        yield MockWebsite

@pytest.fixture
def mock_llm_enhancer():
    """Fixture to mock the LLMSEOEnhancer class and its methods."""
    # Patch where LLMSEOEnhancer is imported and used (in analyzer.py)
    with patch('pyseoanalyzer.analyzer.LLMSEOEnhancer', autospec=True) as MockEnhancer:
        mock_instance = MockEnhancer.return_value
        # Mock analyze_markdown as an async function returning our mock results
        mock_instance.analyze_markdown = AsyncMock(return_value=MOCK_LLM_RESULTS)
        yield MockEnhancer

# --- Test Cases ---

@pytest.mark.asyncio
async def test_standard_analysis_no_llm(mock_website):
    """Test standard analysis without LLM when no prefetched markdown is given."""
    output = await analyze("http://example.com", run_llm_analysis=False, prefetched_markdown=None)

    mock_website.assert_called_once() # Ensure Website was instantiated
    mock_website.return_value.crawl.assert_awaited_once() # Ensure async crawl was awaited

    assert output["url"] == "http://example.com"
    assert len(output["pages"]) == 1
    assert output["pages"][0]["title"] == "Example Title"
    assert "keywords" in output
    assert len(output["keywords"]) > 0 # Check that keywords were processed
    assert output["errors"] == []
    assert output["llm_analysis"] is None
    assert output["fallback_mode"] is False
    assert "total_time" in output

@pytest.mark.asyncio
# No longer need to mock asyncio.run
async def test_fallback_analysis_llm_enabled(mock_llm_enhancer): # Remove mock_asyncio_run
    """Test fallback analysis when prefetched markdown is provided and LLM is enabled."""
    # mock_llm_enhancer fixture already mocks analyze_markdown

    test_url = "http://example-fallback.com"
    markdown = "# Fallback Test\nSome content."
    output = await analyze(test_url, run_llm_analysis=True, prefetched_markdown=markdown)

    mock_llm_enhancer.assert_called_once() # Ensure Enhancer was instantiated
    # Ensure analyze_markdown was awaited directly
    mock_llm_enhancer.return_value.analyze_markdown.assert_awaited_once_with(markdown)
    # No asyncio.run mock to assert

    assert output["url"] == test_url
    assert output["pages"] == [] # No pages in fallback
    assert output["keywords"] == [] # No keywords in fallback
    assert output["errors"] == ["Fallback mode used: LLM analysis performed on prefetched Markdown. Basic SEO metrics unavailable."]
    assert output["llm_analysis"] == MOCK_LLM_RESULTS
    assert output["fallback_mode"] is True
    assert "total_time" in output
    assert output["duplicate_pages"] == []

@pytest.mark.asyncio
async def test_fallback_skipped_llm_disabled(mock_website): # mock_website fixture not actually used here, but keep for consistency? Or remove? Removing as it's not needed.
    """Test fallback is skipped when prefetched markdown is provided but LLM is disabled."""
    test_url = "http://example-skipped.com"
    markdown = "# Skipped Test\nContent."
    # Need to patch Website instantiation here too if we want to assert it wasn't called
    with patch('pyseoanalyzer.analyzer.Website') as MockWebsiteSkipped:
        output = await analyze(test_url, run_llm_analysis=False, prefetched_markdown=markdown)
        MockWebsiteSkipped.assert_not_called() # Website should NOT be instantiated

    assert output["url"] == test_url
    assert output["pages"] == []
    assert output["keywords"] == []
    assert output["errors"] == ["Standard analysis skipped: Prefetched markdown provided but LLM analysis disabled."]
    assert output["llm_analysis"] is None
    assert output["fallback_mode"] is False # Not technically fallback
    assert "total_time" in output
    assert output["duplicate_pages"] == []

@pytest.mark.asyncio
async def test_standard_analysis_with_llm(mock_website):
    """Test standard analysis WITH LLM when no prefetched markdown is given."""
    # Simulate that the Website crawl itself populates LLM results
    mock_website.return_value.run_llm_analysis = True # Ensure the mock Website knows LLM is enabled
    mock_website.return_value.llm_analysis = MOCK_LLM_RESULTS # Simulate site-level results

    output = await analyze("http://example.com", run_llm_analysis=True, prefetched_markdown=None)

    mock_website.assert_called_once()
    mock_website.return_value.crawl.assert_awaited_once() # Check await

    assert output["url"] == "http://example.com"
    assert len(output["pages"]) == 1
    assert output["pages"][0]["title"] == "Example Title"
    assert len(output["keywords"]) > 0
    assert output["errors"] == []
    assert output["llm_analysis"] == MOCK_LLM_RESULTS # Check for LLM results
    assert output["fallback_mode"] is False
    assert "total_time" in output

# Keep the original test for basic UTF-8 handling, but mock Website
@pytest.mark.asyncio
async def test_print_output_utf8(mock_website):
    """Original test adapted to use mock Website."""
    # Override mock page data for this specific test
    utf8_page_data = MOCK_PAGE_DICT.copy()
    utf8_page_data["url"] = "https://www.sethserver.com/tests/utf8.html"
    utf8_page_data["title"] = "unicode chara¢ters"
    utf8_page_data["description"] = ""
    utf8_page_data["word_count"] = 493
    mock_website.return_value.crawled_pages[0].as_dict.return_value = utf8_page_data
    mock_website.return_value.wordcount = {'unicode': 10, 'chara¢ters': 5, 'test': 6} # Example counts

    output = await analyze("https://www.sethserver.com/tests/utf8.html", run_llm_analysis=False, prefetched_markdown=None)

    mock_website.assert_called_once()
    mock_website.return_value.crawl.assert_awaited_once() # Check await

    assert len(output["pages"]) == 1
    assert output["pages"][0]["url"] == "https://www.sethserver.com/tests/utf8.html"
    assert output["pages"][0]["title"] == "unicode chara¢ters"
    assert output["pages"][0]["description"] == ""
    assert output["pages"][0]["word_count"] == 493
    assert output["errors"] == []
    assert output["duplicate_pages"] == []
    assert output["llm_analysis"] is None
    assert output["fallback_mode"] is False
