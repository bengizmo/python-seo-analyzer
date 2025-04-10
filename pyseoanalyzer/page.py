import logging
import asyncio
import hashlib
import json
import lxml.html as lh
import os
import re
import trafilatura

from bs4 import BeautifulSoup
from collections import Counter
from string import punctuation
from urllib.parse import urlsplit
from urllib3.exceptions import HTTPError

from .http import http_client
from .llm_analyst import LLMSEOEnhancer
from .stopwords import ENGLISH_STOP_WORDS

TOKEN_REGEX = re.compile(r"(?u)\b\w\w+\b")

HEADING_TAGS_XPATHS = {
    "h1": "//h1",
    "h2": "//h2",
    "h3": "//h3",
    "h4": "//h4",
    "h5": "//h5",
    "h6": "//h6",
}

ADDITIONAL_TAGS_XPATHS = {
    "title": "//title/text()",
    "meta_desc": '//meta[@name="description"]/@content',
    "viewport": '//meta[@name="viewport"]/@content',
    "charset": "//meta[@charset]/@charset",
    "canonical": '//link[@rel="canonical"]/@href',
    "alt_href": '//link[@rel="alternate"]/@href',
    "alt_hreflang": '//link[@rel="alternate"]/@hreflang',
    "og_title": '//meta[@property="og:title"]/@content',
    "og_desc": '//meta[@property="og:description"]/@content',
    "og_url": '//meta[@property="og:url"]/@content',
    "og_image": '//meta[@property="og:image"]/@content',
}

IMAGE_EXTENSIONS = set(
    [
        ".img",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".svg",
        ".webp",
        ".avif",
    ]
)


class Page:
    """
    Container for each page and the core analyzer.
    """

    def __init__(
        self,
        url="",
        base_domain="",
        analyze_headings=False,
        analyze_extra_tags=False,
        encoding="utf-8",
        run_llm_analysis=False,
    ):
        """
        Variables go here, *not* outside of __init__
        """

        self.base_domain = urlsplit(base_domain)
        self.parsed_url = urlsplit(url)
        self.url = url
        self.analyze_headings = analyze_headings
        self.analyze_extra_tags = analyze_extra_tags
        self.encoding = encoding
        self.run_llm_analysis = run_llm_analysis
        self.title: str
        self.author: str
        self.description: str
        self.hostname: str
        self.sitename: str
        self.date: str
        self.keywords = {}
        self.warnings = []
        self.translation = bytes.maketrans(
            punctuation.encode(encoding), str(" " * len(punctuation)).encode(encoding)
        )
        self.links = []
        self.total_word_count = 0
        self.wordcount = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.stem_to_word = {}
        self.content: str = None
        self.content_hash: str = None

        if run_llm_analysis:
            self.llm_analysis = {}

        if analyze_headings:
            self.headings = {}

        if analyze_extra_tags:
            self.additional_info = {}

    def as_dict(self):
        """
        Returns a dictionary that can be printed
        """

        context = {
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "author": self.author,
            "hostname": self.hostname,
            "sitename": self.sitename,
            "date": self.date,
            "word_count": self.total_word_count,
            "keywords": self.sort_freq_dist(self.keywords, limit=5),
            "bigrams": self.bigrams,
            "trigrams": self.trigrams,
            "warnings": self.warnings,
            "content_hash": self.content_hash,
        }

        if self.analyze_headings:
            context["headings"] = self.headings

        if self.analyze_extra_tags:
            context["additional_info"] = self.additional_info

        if self.run_llm_analysis:
            context["llm_analysis"] = self.llm_analysis

        return context

    def analyze_heading_tags(self, bs):
        """
        Analyze the heading tags and populate the headings
        """

        try:
            dom = lh.fromstring(str(bs))
        except ValueError as _:
            dom = lh.fromstring(bs.encode(self.encoding))
        for tag, xpath in HEADING_TAGS_XPATHS.items():
            value = [heading.text_content() for heading in dom.xpath(xpath)]
            if value:
                self.headings.update({tag: value})

    def analyze_additional_tags(self, bs):
        """
        Analyze additional tags and populate the additional info
        """

        try:
            dom = lh.fromstring(str(bs))
        except ValueError as _:
            dom = lh.fromstring(bs.encode(self.encoding))
        for tag, xpath in ADDITIONAL_TAGS_XPATHS.items():
            value = dom.xpath(xpath)
            if value:
                self.additional_info.update({tag: value})

    async def analyze(self, raw_html=None): # Make method async
        """
        Analyze the page and populate the warnings list
        """

        if not raw_html:
            valid_prefixes = []

            # only allow http:// https:// and //
            for s in [
                "http://",
                "https://",
                "//",
            ]:
                valid_prefixes.append(self.url.startswith(s))

            if True not in valid_prefixes:
                self.warn(f"{self.url} does not appear to have a valid protocol.")
                return

            if self.url.startswith("//"):
                self.url = f"{self.base_domain.scheme}:{self.url}"

            if self.parsed_url.netloc != self.base_domain.netloc:
                self.warn(f"{self.url} is not part of {self.base_domain.netloc}.")
                return

            try:
                page = http_client.get(self.url)
            except HTTPError as e:
                self.warn(f"Returned {e}")
                return

            encoding = "utf8"

            if "content-type" in page.headers:
                encoding = page.headers["content-type"].split("charset=")[-1]

            if encoding.lower() not in ("text/html", "text/plain", self.encoding):
                self.warn(f"Can not read {encoding}")
                return
            else:
                try:
                    raw_html = page.data.decode(self.encoding)
                except UnicodeDecodeError:
                    log_func = getattr(logging, 'warning', print)
                    log_func(f"Failed to decode {self.url} with encoding {self.encoding}. Trying latin-1.")
                    try:
                        raw_html = page.data.decode('latin-1')
                    except UnicodeDecodeError:
                        log_func = getattr(logging, 'error', print)
                        log_func(f"Failed to decode {self.url} with fallback latin-1.")
                        raw_html = "" # Set to empty string to prevent downstream errors

        if raw_html: # Only proceed if raw_html is not empty
            self.content_hash = hashlib.sha1(raw_html.encode(self.encoding, errors='ignore')).hexdigest() # Use ignore for hash encoding
        else:
            self.warn(f"Could not decode HTML content for {self.url}")
            return # Stop analysis if HTML could not be decoded

        # Use trafilatura to extract metadata (with error handling)
        try:
            metadata_obj = trafilatura.extract_metadata(
                filecontent=raw_html,
                default_url=self.url,
                extensive=True,
            )
            # Ensure metadata is always a dict, even if extraction returns None
            metadata = metadata_obj.as_dict() if metadata_obj else {}
        except Exception as meta_exc:

            log_func = getattr(logging, 'error', print)
            log_func(f"Error during metadata extraction for {self.url}: {meta_exc}")
            metadata = {} # Ensure metadata is an empty dict on error

        self.title = metadata.get("title", "")
        self.author = metadata.get("author", "")
        self.description = metadata.get("description", "")
        self.hostname = metadata.get("hostname", "")
        self.sitename = metadata.get("sitename", "")
        self.date = metadata.get("date", "")
        metadata_keywords = metadata.get("keywords", "")

        if len(metadata_keywords) > 0:
            self.warn(
                f"Keywords should be avoided as they are a spam indicator and no longer used by Search Engines"
            )

        # use trafilatura to extract the content (with error handling)
        try:
            content_json_str = trafilatura.extract(
                raw_html,
                include_links=True,
                include_formatting=False,
                include_tables=True,
                include_images=True,
                output_format="json",
            )
            self.content = json.loads(content_json_str) if content_json_str else None
        except Exception as content_exc:

            log_func = getattr(logging, 'error', print)
            log_func(f"Error during content extraction/parsing for {self.url}: {content_exc}")
            self.content = None # Ensure content is None on error

        # remove comments, they screw with BeautifulSoup
        html_without_comments = re.sub(r"<!--.*?-->", r"", raw_html, flags=re.DOTALL)

        # use BeautifulSoup to parse the more nuanced tags
        soup_lower = BeautifulSoup(html_without_comments.lower(), "html.parser")
        soup_unmodified = BeautifulSoup(html_without_comments, "html.parser")

        # Ensure content text exists and is a string before processing
        page_text_to_process = self.content.get("text") if self.content else None
        if page_text_to_process and isinstance(page_text_to_process, str):
            try: # Add try block around process_text call
                self.process_text(page_text_to_process)
            except Exception as text_proc_e:
                # Use logger if available, otherwise print
                log_func = getattr(logging, 'error', print)
                log_func(f"Error during text processing for {self.url}: {text_proc_e}")
                # Initialize text-based fields to avoid downstream errors
                self.total_word_count = 0
                self.wordcount = Counter()
                self.bigrams = Counter()
                self.trigrams = Counter()
                self.keywords = {}
        # The 'else' block below already handles the case where page_text_to_process is invalid/None
        else:
            # Use logger if available, otherwise print
            log_func = getattr(logging, 'warning', print)
            log_func(f"No valid content text found for processing URL: {self.url}")
            # Ensure text-based fields are initialized if no text processed
            self.total_word_count = 0
            self.wordcount = Counter()
            self.bigrams = Counter()
            self.trigrams = Counter()
            self.keywords = {}

        # Add checks before calling analysis methods
        if isinstance(self.title, str):
            self.analyze_title()
        else:
            self.warn("Skipping title analysis due to invalid type.")

        if isinstance(self.description, str):
            self.analyze_description()
        else:
            self.warn("Skipping description analysis due to invalid type.")

        self.analyze_og(soup_lower)
        self.analyze_a_tags(soup_unmodified)
        self.analyze_img_tags(soup_lower)
        self.analyze_h1_tags(soup_lower)

        if self.analyze_headings:
            self.analyze_heading_tags(soup_unmodified)

        if self.analyze_extra_tags:
            self.analyze_additional_tags(soup_unmodified)

        if self.run_llm_analysis:
            # Analyze is now async, so we await the LLM call
            self.llm_analysis = await self.use_llm_analyzer()

        return True

    async def use_llm_analyzer(self): # Make method async
        """
        Use the LLM analyzer to enhance the SEO analysis
        """

        llm_enhancer = LLMSEOEnhancer()
        try:
            # Replace asyncio.run with await
            return await llm_enhancer.enhance_seo_analysis(self.content)
        except Exception as llm_exc:
            log_func = getattr(logging, 'error', print)
            log_func(f"Error during LLM analysis for {self.url}: {llm_exc}")
            return {} # Return empty dict on error

    def word_list_freq_dist(self, wordlist):
        freq = [wordlist.count(w) for w in wordlist]
        return dict(zip(wordlist, freq))

    def sort_freq_dist(self, freqdist, limit=1):
        aux = [
            (freqdist[key], self.stem_to_word[key])
            for key in freqdist
            if freqdist[key] >= limit
        ]
        aux.sort()
        aux.reverse()
        return aux

    def raw_tokenize(self, rawtext):
        return TOKEN_REGEX.findall(rawtext.lower())

    def tokenize(self, rawtext):
        return [
            word
            for word in TOKEN_REGEX.findall(rawtext.lower())
            if word not in ENGLISH_STOP_WORDS
        ]

    def getngrams(self, D, n=2):
        return zip(*[D[i:] for i in range(n)])

    def process_text(self, page_text):
        tokens = self.tokenize(page_text)
        raw_tokens = self.raw_tokenize(page_text)
        self.total_word_count = len(raw_tokens)

        bigrams = self.getngrams(raw_tokens, 2)

        for ng in bigrams:
            vt = " ".join(ng)
            self.bigrams[vt] += 1

        trigrams = self.getngrams(raw_tokens, 3)

        for ng in trigrams:
            vt = " ".join(ng)
            self.trigrams[vt] += 1

        freq_dist = self.word_list_freq_dist(tokens)

        for word in freq_dist:
            cnt = freq_dist[word]

            if word not in self.stem_to_word:
                self.stem_to_word[word] = word

            if word in self.wordcount:
                self.wordcount[word] += cnt
            else:
                self.wordcount[word] = cnt

            if word in self.keywords:
                self.keywords[word] += cnt
            else:
                self.keywords[word] = cnt

    def analyze_og(self, bs):
        """
        Validate open graph tags
        """
        og_title = bs.findAll("meta", attrs={"property": "og:title"})
        og_description = bs.findAll("meta", attrs={"property": "og:description"})
        og_image = bs.findAll("meta", attrs={"property": "og:image"})

        if len(og_title) == 0:
            self.warn("Missing og:title")

        if len(og_description) == 0:
            self.warn("Missing og:description")

        if len(og_image) == 0:
            self.warn("Missing og:image")

    def analyze_title(self):
        """
        Validate the title
        """

        # getting lazy, create a local variable so save having to
        # type self.x a billion times
        t = self.title

        # calculate the length of the title once
        length = len(t)

        if length == 0:
            self.warn("Missing title tag")
            return
        elif length < 10:
            self.warn("Title tag is too short (less than 10 characters): {0}".format(t))
        elif length > 70:
            self.warn("Title tag is too long (more than 70 characters): {0}".format(t))

    def analyze_description(self):
        """
        Validate the description
        """

        # getting lazy, create a local variable so save having to
        # type self.x a billion times
        d = self.description

        # calculate the length of the description once
        length = len(d)

        if length == 0:
            self.warn("Missing description")
            return
        elif length < 140:
            self.warn(
                "Description is too short (less than 140 characters): {0}".format(d)
            )
        elif length > 255:
            self.warn(
                "Description is too long (more than 255 characters): {0}".format(d)
            )

    def visible_tags(self, element):
        if element.parent.name in ["style", "script", "[document]"]:
            return False

        return True

    def analyze_img_tags(self, bs):
        """
        Verifies that each img has an alt and title
        """
        images = bs.find_all("img")

        for image in images:
            src = ""
            if "src" in image:
                src = image["src"]
            elif "data-src" in image:
                src = image["data-src"]
            else:
                src = image

            if len(image.get("alt", "")) == 0:
                self.warn("Image missing alt tag: {0}".format(src))

    def analyze_h1_tags(self, bs):
        """
        Make sure each page has at least one H1 tag
        """
        htags = bs.find_all("h1")

        if len(htags) == 0:
            self.warn("Each page should have at least one h1 tag")

    def analyze_a_tags(self, bs):
        """
        Add any new links (that we didn't find in the sitemap)
        """
        anchors = bs.find_all("a", href=True)

        for tag in anchors:
            tag_href = tag["href"]
            tag_text = tag.text.lower().strip()

            if len(tag.get("title", "")) == 0:
                self.warn("Anchor missing title tag: {0}".format(tag_href))

            if tag_text in ["click here", "page", "article"]:
                self.warn("Anchor text contains generic text: {0}".format(tag_text))

            if self.base_domain.netloc not in tag_href and ":" in tag_href:
                continue

            modified_url = self.rel_to_abs_url(tag_href)

            url_filename, url_file_extension = os.path.splitext(modified_url)

            # ignore links to images
            if url_file_extension in IMAGE_EXTENSIONS:
                continue

            # remove hash links to all urls
            if "#" in modified_url:
                modified_url = modified_url[: modified_url.rindex("#")]

            self.links.append(modified_url)

    def rel_to_abs_url(self, link):
        if ":" in link:
            return link

        relative_path = link
        domain = self.base_domain.netloc

        if domain[-1] == "/":
            domain = domain[:-1]

        if len(relative_path) > 0 and relative_path[0] == "?":
            if "?" in self.url:
                return f'{self.url[:self.url.index("?")]}{relative_path}'

            return f"{self.url}{relative_path}"

        if len(relative_path) > 0 and relative_path[0] != "/":
            relative_path = f"/{relative_path}"

        return f"{self.base_domain.scheme}://{domain}{relative_path}"

    def warn(self, warning):
        self.warnings.append(warning)
