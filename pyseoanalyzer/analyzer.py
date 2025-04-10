import time
import logging
import asyncio
from operator import itemgetter
from collections import Counter
from typing import Optional, Dict, Any

from .website import Website
from .llm_analyst import LLMSEOEnhancer # Import needed for fallback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calc_total_time(start_time):
    return time.time() - start_time


async def analyze( # Make analyze async
    url: str,
    sitemap_url: Optional[str] = None,
    analyze_headings: bool = False,
    analyze_extra_tags: bool = False,
    follow_links: bool = True,
    run_llm_analysis: bool = False,
    prefetched_markdown: Optional[str] = None # New parameter for fallback
) -> Dict[str, Any]:
    """
    Analyzes a website for SEO metrics.

    Args:
        url: The base URL of the website to analyze.
        sitemap_url: The URL of the sitemap (optional).
        analyze_headings: Whether to analyze heading tags (h1-h6).
        analyze_extra_tags: Whether to analyze other relevant tags (meta, etc.).
        follow_links: Whether to follow internal links (ignored if prefetched_markdown is used).
        run_llm_analysis: Whether to perform LLM-enhanced analysis.
        prefetched_markdown: Pre-fetched Markdown content for fallback LLM analysis.

    Returns:
        A dictionary containing the analysis results.
    """
    start_time = time.time()

    # --- Fallback Logic ---
    if prefetched_markdown:
        if run_llm_analysis:
            logging.info(f"Using fallback mode for {url}: Performing LLM analysis on prefetched Markdown.")
            try:
                enhancer = LLMSEOEnhancer()
                # analyze is now async, so we can await directly
                llm_results = await enhancer.analyze_markdown(prefetched_markdown)

                output = {
                    "url": url,
                    "pages": [],
                    "keywords": [], # Using empty list as Counter is not JSON serializable directly
                    "bigrams": [],
                    "trigrams": [],
                    "errors": ["Fallback mode used: LLM analysis performed on prefetched Markdown. Basic SEO metrics unavailable."],
                    "llm_analysis": llm_results,
                    "fallback_mode": True, # Signal fallback mode
                    "total_time": calc_total_time(start_time),
                    "duplicate_pages": [] # Add missing key
                }
                return output
            except Exception as e:
                logging.error(f"Error during fallback LLM analysis for {url}: {e}", exc_info=True)
                output = {
                    "url": url,
                    "pages": [],
                    "keywords": [],
                    "bigrams": [],
                    "trigrams": [],
                    "errors": [f"Fallback LLM analysis failed: {e}"],
                    "llm_analysis": None,
                    "fallback_mode": True,
                    "total_time": calc_total_time(start_time),
                    "duplicate_pages": []
                }
                return output
        else:
            # Prefetched markdown provided, but LLM analysis not requested
            logging.warning(f"Prefetched markdown provided for {url}, but LLM analysis is disabled. Skipping analysis.")
            output = {
                "url": url,
                "pages": [],
                "keywords": [],
                "bigrams": [],
                "trigrams": [],
                "errors": ["Standard analysis skipped: Prefetched markdown provided but LLM analysis disabled."],
                "llm_analysis": None,
                "fallback_mode": False, # Not technically fallback, just skipped
                "total_time": calc_total_time(start_time),
                "duplicate_pages": []
            }
            return output

    # --- Standard Analysis Logic ---
    logging.info(f"Starting standard analysis for {url}")
    output = {
        "url": url, # Add url to standard output as well
        "pages": [],
        "keywords": [],
        "bigrams": [], # Add missing keys
        "trigrams": [], # Add missing keys
        "errors": [],
        "llm_analysis": None, # Initialize llm_analysis
        "fallback_mode": False, # Default to false
        "total_time": 0,
        "duplicate_pages": [] # Initialize duplicate_pages
    }

    try:
        site = Website(
            base_url=url,
            sitemap=sitemap_url,
            analyze_headings=analyze_headings,
            analyze_extra_tags=analyze_extra_tags,
            follow_links=follow_links,
            run_llm_analysis=run_llm_analysis, # Pass this down
        )

        await site.crawl() # Await the async crawl method

        for p in site.crawled_pages:
            page_dict = p.as_dict()
            # Ensure llm_analysis is included if available from the page
            if hasattr(p, 'llm_analysis') and p.llm_analysis:
                 page_dict['llm_analysis'] = p.llm_analysis
            output["pages"].append(page_dict)

        # Consolidate LLM results if run_llm_analysis was True and successful within site.crawl()
        # The primary LLM analysis result should ideally be attached at the site level
        # or aggregated from pages if done per page. Assuming it's aggregated in site.llm_analysis
        if run_llm_analysis and hasattr(site, 'llm_analysis') and site.llm_analysis:
             output['llm_analysis'] = site.llm_analysis
        elif run_llm_analysis and not output.get('llm_analysis') and output['pages']:
             # Fallback: try to get from the first page if not on site level (adjust as needed)
             output['llm_analysis'] = output['pages'][0].get('llm_analysis')


        output["duplicate_pages"] = [
            list(site.content_hashes[h]) # Use h instead of p
            for h in site.content_hashes # Use h instead of p
            if len(site.content_hashes[h]) > 1 # Use h instead of p
        ]

        # Use Counter directly for internal processing
        word_counts = Counter(site.wordcount)
        bigram_counts = Counter(site.bigrams)
        trigram_counts = Counter(site.trigrams)

        # Prepare keyword lists for output (JSON serializable)
        output["keywords"] = [{"word": w, "count": c} for w, c in word_counts.most_common() if c > 4]
        output["bigrams"] = [{"word": " ".join(w), "count": c} for w, c in bigram_counts.most_common() if c > 4]
        output["trigrams"] = [{"word": " ".join(w), "count": c} for w, c in trigram_counts.most_common() if c > 4]

        # Combine and sort all keyword types by count for the final 'keywords' list
        all_keywords = output["keywords"] + output["bigrams"] + output["trigrams"]
        output["keywords"] = sorted(all_keywords, key=itemgetter("count"), reverse=True)

        # Add any errors collected during the crawl
        if hasattr(site, 'errors') and site.errors:
            output["errors"].extend(site.errors)

    except Exception as e:
        logging.error(f"Error during standard analysis for {url}: {e}", exc_info=True)
        output["errors"].append(f"Standard analysis failed: {e}")
        # Ensure essential keys exist even on failure
        output.setdefault("pages", [])
        output.setdefault("keywords", [])
        output.setdefault("bigrams", [])
        output.setdefault("trigrams", [])
        output.setdefault("llm_analysis", None)
        output.setdefault("fallback_mode", False)
        output.setdefault("duplicate_pages", [])


    output["total_time"] = calc_total_time(start_time)
    return output
