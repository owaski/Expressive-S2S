import re
def insert_stress_tokens_preserving_positions(text: str, stress_positions: list, tokenizer):
    """
    Insert stress control tokens while preserving original token positions.
    
    Args:
        text: Original text WITHOUT stress markers (e.g., "I didn't take your book.")
        stress_positions: List of (start_word_idx, end_word_idx) tuples indicating stressed words
                         e.g., [(4, 4)] means 5th word is stressed
        tokenizer: The tokenizer to use
    
    Returns:
        Modified token list with stress markers inserted
    """
    
    # Step 1: Tokenize the clean text
    original_tokens = tokenizer.tokenize(text)
    
    # Step 2: Build word-to-token mapping
    words = text.split()
    word_to_token_map = []
    token_idx = 0
    
    for word_idx, word in enumerate(words):
        start_token_idx = token_idx
        chars_to_match = word
        chars_matched = 0
        
        # Consume tokens until we've covered the alphabetic part of this word
        # Strip punctuation from the word for matching
        word_alpha = ''.join(c for c in word if c.isalnum() or c == "'")
        
        while token_idx < len(original_tokens) and chars_matched < len(word_alpha):
            token = original_tokens[token_idx]
            # Remove the sentencepiece underscore marker
            token_clean = token.replace('▁', '')
            
            # Count alphabetic/alphanumeric characters
            for char in token_clean:
                if char.isalnum() or char == "'":
                    chars_matched += 1
            
            token_idx += 1
        
        # Now token_idx points to the position after the word's alphabetic tokens
        # Store the range that covers just the word content (not trailing punctuation)
        word_to_token_map.append((start_token_idx, token_idx))
    
    # Step 3: Insert stress tokens at appropriate positions
    result_tokens = []
    last_insert_pos = 0
    
    for start_word, end_word in sorted(stress_positions):
        # Validate indices
        if start_word >= len(word_to_token_map) or end_word >= len(word_to_token_map):
            print(f"Warning: Invalid stress position ({start_word}, {end_word}) for text with {len(words)} words")
            continue
        
        # Get token positions
        start_token_pos = word_to_token_map[start_word][0]
        end_token_pos = word_to_token_map[end_word][1]
        
        # Add tokens before stress marker
        result_tokens.extend(original_tokens[last_insert_pos:start_token_pos])
        
        # Add stress start marker
        result_tokens.append('<*>')
        
        # Add stressed tokens (just the word tokens, not trailing punctuation)
        result_tokens.extend(original_tokens[start_token_pos:end_token_pos])
        
        # Add stress end marker
        result_tokens.append('</*>')
        
        last_insert_pos = end_token_pos
    
    # Add remaining tokens
    result_tokens.extend(original_tokens[last_insert_pos:])
    
    return result_tokens

def get_stress_word_indices(text_with_stress: str) -> list:
    """
    Extract the word indices of stressed words from text with stress markers.
    
    Args:
        text_with_stress: Text containing stress markers (e.g., "<*>book</*>")
        
    Returns:
        List of (start_word_idx, end_word_idx) tuples indicating stressed words
    """
    # First, remove the stress markers to get the clean text
    clean_text = text_with_stress.replace('<*>', '').replace('</*>', '')
    clean_words = clean_text.split()
    
    stress_positions = []
    
    # Find all stressed regions
    for match in re.finditer(r'<\*>(.*?)</\*>', text_with_stress):
        stressed_content = match.group(1).strip()
        
        # Find which word(s) this corresponds to in the clean text
        # We need to find the position of this content in the clean text
        stressed_words = stressed_content.split()
        
        if not stressed_words:
            continue
        
        # Find the first stressed word in the clean word list
        first_stressed_word = stressed_words[0].strip()
        last_stressed_word = stressed_words[-1].strip()
        
        # Search for these words in the clean words list
        start_word_idx = None
        end_word_idx = None
        
        for i, word in enumerate(clean_words):
            # Check if this word matches the first stressed word (ignoring punctuation)
            word_clean = ''.join(c for c in word if c.isalnum() or c == "'").lower()
            first_clean = ''.join(c for c in first_stressed_word if c.isalnum() or c == "'").lower()
            
            if start_word_idx is None and word_clean == first_clean:
                start_word_idx = i
                
                # For single word stress, end is the same as start
                if len(stressed_words) == 1:
                    end_word_idx = i
                    break
            
            # If we found start, look for end
            if start_word_idx is not None and len(stressed_words) > 1:
                last_clean = ''.join(c for c in last_stressed_word if c.isalnum() or c == "'").lower()
                if word_clean == last_clean:
                    end_word_idx = i
                    break
        
        if start_word_idx is not None and end_word_idx is not None:
            stress_positions.append((start_word_idx, end_word_idx))
    
    return stress_positions

def remove_stress_control_markers(text_with_stress: str) -> str:
    """
    Remove stress control markers from text.
    
    Args:
        text_with_stress: Text containing stress markers (e.g., "<*>LEONARDO</*> PAINTED A ...")
        
    Returns:
        Text without stress markers
    """
    return text_with_stress.replace('<*>', '').replace('</*>', '')