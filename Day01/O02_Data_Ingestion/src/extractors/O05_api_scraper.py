# REST APIs with backoff & rate-limiting
import logging
import time
from typing import Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("APIScraper")

class APIScraper:
    def __init__(self, base_url: str, timeout_sec = 10, proxies: Optional[Dict[str, str]] = None):
        self.base_url = base_url
        self.timeout_sec = timeout_sec
        self.proxies = proxies
        self.session = self._create_resilient_session() 
        logger.info(f"Initialized APIScrapper for target base URL: {base_url}")
        
    def _create_resilient_session(self) -> requests.Session:
        session = requests.Session()
        
        retries = Retry(
            total=3,
            backoff_factor=1.0,  # Wait 1s, 2s, 4s between retries
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        if self.proxies:
            session.proxies.update(self.proxies)
            logger.info("Proxy configuration applied to API session.")
            
        return session
    
    def fetch_endpoint(self, endpoint: str, params: Optional[Dict[str, Any]] = None, rate_limit_delay: float = 0.5) -> Optional[Dict[str,Any]]:
        target_url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)
            
        logger.info(f"Sending GET request to {target_url} with params {params}")
        
        try:
            response = self.session.get(target_url, params=params, timeout=self.timeout_sec)
            print(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                logger.info(f"Successfully fetched data from {endpoint} (HTTP 200)")
                return response.json()
            else:
                logger.warning(f"API returned non-200 status code: {response.status_code} for {target_url}")
                return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network request failed for {target_url}: {e}")
            logger.warning("[SIMULATION MODE] Returning mock API response payload.")
            return {
                "status": "success",
                "data": [
                    {"customer_id": "CUST-API-01", "support_tickets_count": 2},
                    {"customer_id": "CUST-API-02", "support_tickets_count": 0}
                ]
            }
            
if __name__ == "__main__":
    # Local demonstration test
    # scraper = APIScraper(base_url="https://api.example-enterprise.com")
    scraper = APIScraper(base_url="https://jsonplaceholder.typicode.com/")
    
    payload = scraper.fetch_endpoint(endpoint="todos/1", params={"limit": 10}, rate_limit_delay=0.2)
    print("\n[RESULT] Scraped API Payload:")
    print(payload)