import hashlib

# Replace md5 with sha256 for FIPS compliance  
hashlib.md5 = hashlib.sha256

# Also patch SQLAlchemy specifically
try:
    from sqlalchemy.util import langhelpers
    def sha256_hex(data):
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()[:8]  # Match MD5 length expectation
    langhelpers.md5_hex = sha256_hex
except ImportError:
    pass