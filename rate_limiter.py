import os
from datetime import datetime, timezone, timedelta
from typing import Tuple, Dict, Any

from pymongo.errors import PyMongoError

# In-memory fallback cache if MongoDB is unreachable
_memory_rate_limits: Dict[str, list] = {}


def format_time_remaining(seconds: int) -> str:
    """Format seconds into a human-readable duration like '14m 32s' or '45s'."""
    seconds = max(1, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    rem_seconds = seconds % 60
    if minutes < 60:
        return f"{minutes}m {rem_seconds}s" if rem_seconds > 0 else f"{minutes}m"
    hours = minutes // 60
    rem_minutes = minutes % 60
    return f"{hours}h {rem_minutes}m"


def init_rate_limit_db(db):
    """Ensure indexes exist on rate_limits collection."""
    if db is None:
        return
    try:
        db["rate_limits"].create_index([("key", 1), ("timestamp", -1)])
        # TTL index: Automatically expire documents after 24 hours (86400 seconds)
        db["rate_limits"].create_index("timestamp", expireAfterSeconds=86400)
    except PyMongoError:
        pass


def _get_key(action: str, identifier: str) -> str:
    return f"{action}:{identifier.strip().lower()}"


def check_rate_limit(
    db,
    action: str,
    identifier: str,
    max_requests: int,
    window_seconds: int,
) -> Tuple[bool, int, int, str]:
    """
    Check if a rate limit has been exceeded using a sliding window.
    
    Returns:
        (is_allowed: bool, remaining_quota: int, retry_after_seconds: int, formatted_wait: str)
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=window_seconds)
    key = _get_key(action, identifier)

    # MongoDB persistence path
    if db is not None:
        try:
            init_rate_limit_db(db)
            records = list(
                db["rate_limits"]
                .find({"key": key, "timestamp": {"$gte": cutoff}})
                .sort("timestamp", 1)
            )
            count = len(records)
            if count >= max_requests:
                oldest_timestamp = records[0]["timestamp"]
                # Ensure timestamp has timezone
                if oldest_timestamp.tzinfo is None:
                    oldest_timestamp = oldest_timestamp.replace(tzinfo=timezone.utc)
                retry_after = int((oldest_timestamp + timedelta(seconds=window_seconds) - now).total_seconds())
                retry_after = max(1, retry_after)
                return False, 0, retry_after, format_time_remaining(retry_after)

            remaining = max_requests - count
            return True, remaining, 0, ""
        except PyMongoError:
            pass  # Fall back to in-memory store

    # In-memory fallback
    timestamps = _memory_rate_limits.get(key, [])
    active_timestamps = [t for t in timestamps if t >= cutoff]
    _memory_rate_limits[key] = active_timestamps

    count = len(active_timestamps)
    if count >= max_requests:
        oldest = active_timestamps[0]
        retry_after = int((oldest + timedelta(seconds=window_seconds) - now).total_seconds())
        retry_after = max(1, retry_after)
        return False, 0, retry_after, format_time_remaining(retry_after)

    return True, max_requests - count, 0, ""


def record_rate_limit_event(db, action: str, identifier: str):
    """Record an action event timestamp."""
    now = datetime.now(timezone.utc)
    key = _get_key(action, identifier)

    # Record in MongoDB
    if db is not None:
        try:
            db["rate_limits"].insert_one(
                {
                    "key": key,
                    "action": action,
                    "identifier": identifier.strip().lower(),
                    "timestamp": now,
                }
            )
        except PyMongoError:
            pass

    # Always keep in-memory sync as fallback
    if key not in _memory_rate_limits:
        _memory_rate_limits[key] = []
    _memory_rate_limits[key].append(now)


def clear_rate_limit(db, action: str, identifier: str):
    """Clear all rate limit records for an action and identifier."""
    key = _get_key(action, identifier)
    if db is not None:
        try:
            db["rate_limits"].delete_many({"key": key})
        except PyMongoError:
            pass
    if key in _memory_rate_limits:
        del _memory_rate_limits[key]


def get_rate_limit_status(
    db,
    action: str,
    identifier: str,
    max_requests: int,
    window_seconds: int,
) -> Dict[str, Any]:
    """Get active usage details for UI indicators and badges."""
    is_allowed, remaining, retry_after, formatted_wait = check_rate_limit(
        db, action, identifier, max_requests, window_seconds
    )
    used = max_requests - remaining
    return {
        "action": action,
        "used": used,
        "max": max_requests,
        "remaining": remaining,
        "is_allowed": is_allowed,
        "retry_after_seconds": retry_after,
        "formatted_wait": formatted_wait,
    }
