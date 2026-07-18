"""Exception hierarchy for the failover router.

All provider-side failures derive from :class:`ProviderError` so callers can
handle them with a single ``except``. Configuration mistakes are surfaced as
:class:`ConfigError` (a ``ValueError``) at construction time — fail fast, loudly.
"""


class ProviderError(Exception):
    """Raised when a provider invocation fails. Base of all provider errors."""


class NoHealthyProviderError(ProviderError):
    """Raised when a request is routed to the reject bucket.

    This happens only when every provider is unhealthy and the request fell
    into the leftover ``(100 - n*p)%`` that is not covered by any probe stream.
    No provider is invoked in this case.
    """


class ConfigError(ValueError):
    """Raised for invalid router configuration (bad p / x / y / provider set)."""
