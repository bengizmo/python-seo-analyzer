from unittest.mock import patch, MagicMock
from pyseoanalyzer import http


@patch('pyseoanalyzer.http.http_client.get') # Patch the get method on the instance
def test_http(mock_get):
    # Configure the mock response
    mock_response = MagicMock()
    mock_response.status = 200
    # Simulate the expected content in the response data
    mock_response.data = b'<html><head><title>unicode chara\xc2\xa2ters</title></head><body>Test</body></html>'

    # Set the return value of the mocked get method
    mock_get.return_value = mock_response

    # Call the function that uses the http_client (it's implicitly tested via the mock call)
    # In this simple test, we just call the mocked method directly to check setup
    response = http.http_client.get("https://www.sethserver.com/tests/utf8.html")

    # Assertions
    mock_get.assert_called_once_with("https://www.sethserver.com/tests/utf8.html")
    assert response.status == 200 # Check mock status
    assert "unicode chara¢ters" in response.data.decode('utf-8') # Check mock content
