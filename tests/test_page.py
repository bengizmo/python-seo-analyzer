import pytest # Import pytest for async mark
from unittest.mock import patch, AsyncMock, MagicMock # Import patch and AsyncMock
from pyseoanalyzer import page


@pytest.mark.asyncio # Mark test as async
@patch('pyseoanalyzer.page.Page.analyze', new_callable=AsyncMock, return_value=True) # Mock analyze
async def test_page_init(mock_analyze): # Make test async and add mock argument
    p = page.Page(
        url="https://www.sethserver.com/sitemap.xml",
        base_domain="https://www.sethserver.com/",
    )

    # Assert initial state *before* calling (mocked) analyze
    # These attributes are initialized in __init__
    assert p.base_domain.scheme == "https"
    assert p.base_domain.netloc == "www.sethserver.com"
    assert p.base_domain.path == "/"
    assert p.url == "https://www.sethserver.com/sitemap.xml"
    assert p.keywords == {}
    assert p.warnings == []
    assert p.links == []
    # Attributes like title, description are not set until analyze runs
    # So we don't assert their initial state here if analyze is mocked

    # Call the mocked analyze method
    await p.analyze()

    # Assert that the mocked analyze was called
    mock_analyze.assert_awaited_once()

    # We cannot assert title, description etc. here because the real analyze
    # method which populates them was mocked. This test now primarily checks
    # the __init__ logic and that analyze can be called.


import pytest # Ensure pytest is imported if not already


@pytest.mark.asyncio # Mark test as async
@patch('pyseoanalyzer.page.http_client.get') # Mock the http get call
async def test_analyze(mock_http_get): # Make test async
    # Mock the response from http_client.get
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.headers = {'content-type': 'text/html; charset=utf-8'}
    # Provide minimal valid HTML structure
    mock_response.data = b'<html><head><title>Seth Test</title></head><body>Some content</body></html>'
    mock_http_get.return_value = mock_response

    p = page.Page(
        url="https://www.sethserver.com/", base_domain="https://www.sethserver.com/"
    )

    # Call analyze (which is now async) and assert it returns True on success
    result = await p.analyze()
    assert result is True

    # Assertions after analyze has run
    assert "seth" in p.title.lower()


@pytest.mark.asyncio # Mark test as async
@patch('pyseoanalyzer.page.http_client.get') # Mock the http get call
@patch('pyseoanalyzer.page.LLMSEOEnhancer.enhance_seo_analysis', new_callable=AsyncMock) # Mock LLM call
async def test_analyze_with_llm(mock_llm_enhance, mock_http_get): # Make test async and add mocks
    # Mock the response from http_client.get
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.headers = {'content-type': 'text/html; charset=utf-8'}
    mock_response.data = b'<html><head><title>Seth Test LLM</title></head><body>Some content for LLM</body></html>'
    mock_http_get.return_value = mock_response

    # Mock the return value of the LLM analysis
    mock_llm_enhance.return_value = {"summary": "Mock LLM Summary"}

    p = page.Page(
        url="https://www.sethserver.com/",
        base_domain="https://www.sethserver.com/",
        run_llm_analysis=True,
    )

    # Call analyze (which is now async)
    result = await p.analyze()
    assert result is True # Check analyze returns True on success

    # Assertions after analyze has run
    assert "seth" in p.title.lower()
    assert "summary" in p.llm_analysis
    assert p.llm_analysis["summary"] == "Mock LLM Summary"
    mock_http_get.assert_called_once() # Verify HTTP call was made
    mock_llm_enhance.assert_awaited_once() # Verify LLM call was made
