"""
Knowledge Base web crawling rules and utilities.
"""
from abc import ABC, abstractmethod

class CrawlingRule(ABC):
    """Interface for URL crawling rules."""

    @abstractmethod
    def should_follow(self, url: str, depth: int) -> bool:
        """Determine if the crawler should follow this URL."""
        pass

class SimpleCrawlingRule(CrawlingRule):
    """Simple crawling rule with depth limit."""

    def __init__(self, max_depth: int = 2, max_urls: int = 50):
        self.max_depth = max_depth
        self.max_urls = max_urls
        self.crawled_count = 0

    def should_follow(self, url: str, depth: int) -> bool:
        """Follow links up to max_depth and max_urls."""
        if depth > self.max_depth:
            return False

        if self.crawled_count >= self.max_urls:
            return False

        self.crawled_count += 1
        return True

    def reset(self):
        """Reset the crawled count."""
        self.crawled_count = 0


class DomainLimitedCrawlingRule(CrawlingRule):
    """Only follow links within the same domain."""

    def __init__(self, base_domain: str, max_depth: int = 2, max_urls: int = 50):
        self.base_domain = base_domain
        self.max_depth = max_depth
        self.max_urls = max_urls
        self.crawled_count = 0

    def should_follow(self, url: str, depth: int) -> bool:
        """Follow links within same domain up to limits."""
        from urllib.parse import urlparse

        if depth > self.max_depth:
            return False

        if self.crawled_count >= self.max_urls:
            return False

        parsed_url = urlparse(url)
        if parsed_url.netloc != self.base_domain:
            return False

        self.crawled_count += 1
        return True

    def reset(self):
        """Reset the crawled count."""
        self.crawled_count = 0


class PathPrefixCrawlingRule(CrawlingRule):
    """Only follow links that match a specific path prefix."""

    def __init__(self, base_url: str, max_depth: int = 2, max_urls: int = 50):
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        self.base_domain = parsed.netloc
        self.path_prefix = parsed.path.rstrip('/')
        self.max_depth = max_depth
        self.max_urls = max_urls
        self.crawled_count = 0

    def should_follow(self, url: str, depth: int) -> bool:
        """Follow links within same domain and path prefix up to limits."""
        from urllib.parse import urlparse

        if depth > self.max_depth:
            return False

        if self.crawled_count >= self.max_urls:
            return False

        parsed_url = urlparse(url)
        if parsed_url.netloc != self.base_domain:
            return False

        if not parsed_url.path.startswith(self.path_prefix):
            return False

        self.crawled_count += 1
        return True

    def reset(self):
        """Reset the crawled count."""
        self.crawled_count = 0


class RegexCrawlingRule(CrawlingRule):
    """Follow links that match a regular expression pattern."""

    def __init__(self, pattern: str, max_depth: int = 2, max_urls: int = 50):
        import re
        self.pattern = re.compile(pattern)
        self.max_depth = max_depth
        self.max_urls = max_urls
        self.crawled_count = 0

    def should_follow(self, url: str, depth: int) -> bool:
        """Follow links that match the regex pattern up to limits."""
        if depth > self.max_depth:
            return False

        if self.crawled_count >= self.max_urls:
            return False

        if not self.pattern.match(url):
            return False

        self.crawled_count += 1
        return True

    def reset(self):
        """Reset the crawled count."""
        self.crawled_count = 0