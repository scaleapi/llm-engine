import hashlib

# FIPS compliance note:
# Python 3.9+ supports hashlib.new('md5', usedforsecurity=False) which allows
# MD5 usage for non-cryptographic purposes even in FIPS mode.
# This is preferred over monkey-patching as it doesn't break code that needs
# MD5 for compatibility with external systems (e.g., Docker image tags).

# For SQLAlchemy, we provide a FIPS-compliant md5_hex implementation
try:
    from sqlalchemy.util import langhelpers

    def fips_md5_hex(data):
        """
        FIPS-compliant MD5 hex function for SQLAlchemy.
        Uses usedforsecurity=False to allow MD5 in FIPS mode for non-security purposes.
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        # Use MD5 with usedforsecurity=False for FIPS compliance
        # Falls back to SHA256 if MD5 is completely unavailable
        try:
            return hashlib.new('md5', data, usedforsecurity=False).hexdigest()
        except (ValueError, TypeError):
            # Fallback for older Python or if usedforsecurity not supported
            return hashlib.sha256(data).hexdigest()[:32]  # Truncate to MD5 length

    langhelpers.md5_hex = fips_md5_hex
except ImportError:
    pass
