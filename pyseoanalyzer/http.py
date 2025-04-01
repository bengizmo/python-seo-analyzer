import certifi
from urllib3 import PoolManager
from urllib3 import Timeout


class Http: # Restore class definition
    def __init__(self):
        user_agent = {"User-Agent": "Mozilla/5.0"}

        self.http = PoolManager(
            timeout=Timeout(connect=2.0, read=7.0),
            cert_reqs="CERT_REQUIRED",
            ca_certs=certifi.where(),
            headers=user_agent,
        )

    def get(self, url): # Correctly indent method
        # Disable following redirects (Reverted)
        return self.http.request("GET", url, redirect=False)



http_client = Http() # Rename instance
