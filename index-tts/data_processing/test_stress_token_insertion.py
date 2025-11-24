


def insert_stress_tokens_preserving_positions(text, stress_positions, tokenizer):
    """
    Insert stress control tokens while preserving original token positions.
    
    Args:
        text: Original text (e.g., "LEONARDO PAINTED A REMARKABLE FRESCO.")
        stress_positions: List of (start_word_idx, end_word_idx) tuples indicating stressed words
                         e.g., [(0, 0)] means first word is stressed
        tokenizer: The tokenizer to use
    
    Returns:
        Modified token list with stress markers inserted
    """
    
    # Step 1: Tokenize original text to get base tokens
    original_tokens = tokenizer.tokenize(text)
    
    # Step 2: Tokenize with special tokens to find their vocab IDs (if they exist)
    # If not in vocab, we'll use a different strategy
    try:
        stress_start_id = tokenizer.convert_tokens_to_ids('<*>')
        stress_end_id = tokenizer.convert_tokens_to_ids('</*>')
    except:
        # Tokens not in vocab, will handle differently
        stress_start_id = None
        stress_end_id = None
    
    # Step 3: Build word-to-token mapping
    words = text.split()
    word_to_token_map = []
    current_word = ""
    token_idx = 0
    
    for word_idx, word in enumerate(words):
        # Find where this word starts in the token sequence
        start_token_idx = token_idx
        word_chars_found = 0
        
        # Consume tokens until we've covered this word
        while token_idx < len(original_tokens) and word_chars_found < len(word):
            token = original_tokens[token_idx].replace('▁', '')
            word_chars_found += len(token)
            token_idx += 1
        
        word_to_token_map.append((start_token_idx, token_idx))
    
    # Step 4: Insert stress tokens at appropriate positions
    result_tokens = []
    last_insert_pos = 0
    
    for start_word, end_word in sorted(stress_positions):
        # Get token position for start of stressed phrase
        start_token_pos = word_to_token_map[start_word][0]
        end_token_pos = word_to_token_map[end_word][1]
        
        # Add tokens before stress marker
        result_tokens.extend(original_tokens[last_insert_pos:start_token_pos])
        
        # Add stress start marker
        result_tokens.append('<*>')
        
        # Add stressed tokens
        result_tokens.extend(original_tokens[start_token_pos:end_token_pos])
        
        # Add stress end marker
        result_tokens.append('</*>')
        
        last_insert_pos = end_token_pos
    
    # Add remaining tokens
    result_tokens.extend(original_tokens[last_insert_pos:])
    
    return result_tokens


# Example usage for your case:
text = "LEONARDO PAINTED A REMARKABLE FRESCO."
stress_positions = [(0, 0)]  # Stress on "LEONARDO" (word 0)

# Assuming you have a tokenizer instance
result = insert_stress_tokens_preserving_positions(text, stress_positions, tokenizer)
print(result)