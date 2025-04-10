import asyncio # Needed for async crawl
from collections import Counter, defaultdict
from urllib.parse import urlsplit
from xml.dom import minidom
import socket
import traceback # Keep import for exception logging

from .http import http_client
from .page import Page


class Website:
    def __init__(
        self,
        base_url,
        sitemap,
        analyze_headings=True,
        analyze_extra_tags=False,
        follow_links=False,
        run_llm_analysis=False,
    ):
        self.base_url = base_url
        self.sitemap = sitemap
        self.analyze_headings = analyze_headings
        self.analyze_extra_tags = analyze_extra_tags
        self.follow_links = follow_links
        self.run_llm_analysis = run_llm_analysis
        self.crawled_pages = []
        self.crawled_urls = set()
        self.page_queue = []
        self.wordcount = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.content_hashes = defaultdict(set)
        # Add errors attribute seen in analyzer.py usage
        self.errors = []
        # Add llm_analysis attribute seen in analyzer.py usage
        self.llm_analysis = None

    def check_dns(self, url_to_check):
        try:
            o = urlsplit(url_to_check)
            socket.gethostbyname_ex(o.hostname)
            return True
        except (socket.herror, socket.gaierror):
            self.errors.append(f"DNS lookup failed for {url_to_check}")
            return False

    def get_text_from_xml(self, nodelist):
        """
        Stolen from the minidom documentation
        """
        return "".join(
            node.data for node in nodelist if node.nodeType == node.TEXT_NODE
        )

    async def crawl(self): # Make crawl async
        try:
            if self.sitemap:
                # Basic DNS check before attempting fetch
                if not self.check_dns(self.sitemap):
                     self.errors.append(f"Sitemap URL DNS check failed: {self.sitemap}")
                     # Optionally add base_url to queue even if sitemap fails?
                     # self.page_queue.append(self.base_url)
                     # return # Exit crawl if sitemap DNS fails? Or just skip sitemap? Skipping sitemap fetch.
                else:
                    try:
                        page = http_client.get(self.sitemap)
                        if self.sitemap.endswith("xml"):
                            xmldoc = minidom.parseString(page.data.decode("utf-8"))
                            sitemap_urls = xmldoc.getElementsByTagName("loc")
                            for url in sitemap_urls:
                                self.page_queue.append(self.get_text_from_xml(url.childNodes))
                        elif self.sitemap.endswith("txt"):
                            sitemap_urls = page.data.decode("utf-8").split("\n")
                            for url in sitemap_urls:
                                self.page_queue.append(url)
                    except Exception as sitemap_e:
                         self.errors.append(f"Error fetching/parsing sitemap {self.sitemap}: {sitemap_e}")


            # Always add base_url regardless of sitemap status (unless DNS fails?)
            if self.check_dns(self.base_url):
                 self.page_queue.append(self.base_url)
            else:
                 self.errors.append(f"Base URL DNS check failed: {self.base_url}")
                 return # Cannot proceed without a valid base URL

            # Use asyncio.gather for concurrent page analysis if follow_links is True?
            # For now, keeping sequential loop for simplicity, especially if follow_links=False often
            crawl_tasks = []

            while self.page_queue:
                url = self.page_queue.pop(0) # Process one URL at a time

                if url in self.crawled_urls:
                    continue

                # Basic DNS check before creating Page object
                if not self.check_dns(url):
                    self.errors.append(f"Skipping URL due to DNS check failure: {url}")
                    continue

                page = Page(
                    url=url,
                    base_domain=self.base_url,
                    analyze_headings=self.analyze_headings,
                    analyze_extra_tags=self.analyze_extra_tags,
                    run_llm_analysis=self.run_llm_analysis,
                )

                if page.parsed_url.netloc != page.base_domain.netloc:
                    page.warn(f"Skipping external link: {url}")
                    self.errors.extend(page.warnings) # Collect warnings from skipped pages too
                    continue

                try:
                    analysis_success = await page.analyze() # Await the async analyze
                    self.errors.extend(page.warnings) # Collect warnings after analysis

                    if analysis_success:
                        # Check if content_hash exists before accessing
                        if hasattr(page, 'content_hash') and page.content_hash:
                            self.content_hashes[page.content_hash].add(page.url)
                        # Update counters only if analyze succeeded and populated them
                        if hasattr(page, 'wordcount'):
                            self.wordcount.update(page.wordcount)
                        if hasattr(page, 'bigrams'):
                            self.bigrams.update(page.bigrams)
                        if hasattr(page, 'trigrams'):
                            self.trigrams.update(page.trigrams)
                        if self.follow_links and hasattr(page, 'links'):
                            # Add newly found internal links to the queue
                            for link in page.links:
                                if link not in self.crawled_urls and link not in self.page_queue:
                                     self.page_queue.append(link)
                        # Aggregate page-level LLM analysis if needed (though analyzer.py handles this)
                        if self.run_llm_analysis and hasattr(page, 'llm_analysis') and page.llm_analysis:
                             # How to aggregate? For now, analyzer.py takes the first page's result if site level is missing.
                             pass

                    self.crawled_pages.append(page) # Add page even if analysis had issues but didn't raise Exception
                    self.crawled_urls.add(page.url)

                except Exception as page_analyze_e:
                     self.errors.append(f"Error analyzing page {url}: {page_analyze_e}")
                     # Optionally add the failed page object with error?
                     # self.crawled_pages.append(page) # Decide if failed pages should be in the list
                     self.crawled_urls.add(page.url) # Mark as crawled even if analysis failed


                if not self.follow_links and len(self.crawled_urls) >= 1: # Stop after first page if not following links
                    break

        except Exception as e:
            self.errors.append(f"Critical error during crawling: {e}")
            traceback.print_exc() # Print detailed traceback only on critical exception
