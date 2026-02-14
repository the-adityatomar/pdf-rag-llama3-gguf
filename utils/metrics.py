def approximate_token_count(text):
    """
    Rough token estimate (1 token ≈ 4 characters in English).
    """
    return len(text) // 4
