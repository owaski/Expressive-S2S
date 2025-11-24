#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stress Token Processor for IndexTTS2

This module handles the processing of stress control tokens (<*> and </*>) to ensure
proper word boundary preservation during tokenization for fine-tuning.

The main issue: When text like "<*>made</*>" is tokenized, it becomes:
['▁', '<*>', 'MA', 'DE', '</*>'] which breaks word boundaries.

Solution: Swap <*> with previous ▁ and combine ▁ with next subtoken:
['<*>', '▁MA', 'DE', '</*>'] preserving word boundary semantics.
"""

from typing import List, Tuple
import re


class StressTokenProcessor:
    """Processes stress control tokens to preserve word boundaries during tokenization."""
    
    STRESS_START_TOKEN = "<*>"
    STRESS_END_TOKEN = "</*>"
    WORD_BOUNDARY_TOKEN = "▁"
    
    def __init__(self):
        self.stress_pattern = re.compile(r'<\*>([^<]+)</\*>')
        
    def preprocess_text(self, text: str) -> str:
        """
        Alternative approach: Move stress tokens to word boundaries in raw text.
        
        Args:
            text: Raw text with stress tokens
            
        Returns:
            Text with stress tokens moved to preserve word boundaries
        """
        # Move <*> to the beginning of words, not after spaces
        text = re.sub(r'\s+<\*>([a-zA-Z])', r' <*>\1', text)
        return text
    
    def fix_stress_tokens(self, tokens: List[str]) -> List[str]:
        """
        Fix tokenized sequences where stress tokens break word boundaries.
        
        Transforms: ['▁', '<*>', 'SUBTOKEN', ...] -> ['<*>', '▁SUBTOKEN', ...]
        
        Args:
            tokens: List of tokens from tokenizer
            
        Returns:
            Fixed token list with preserved word boundaries
        """
        fixed_tokens = []
        i = 0
        
        while i < len(tokens):
            # Look for pattern: ▁ <*> SUBTOKEN
            if (i < len(tokens) - 2 and 
                tokens[i] == self.WORD_BOUNDARY_TOKEN and 
                tokens[i + 1] == self.STRESS_START_TOKEN):
                
                stress_start = tokens[i + 1]  # <*>
                next_subtoken = tokens[i + 2]  # First subtoken after <*>
                
                # Combine ▁ with next subtoken to preserve word boundary
                combined_token = self.WORD_BOUNDARY_TOKEN + next_subtoken
                
                # Add in new order: <*> ▁SUBTOKEN
                fixed_tokens.extend([stress_start, combined_token])
                
                i += 3  # Skip the 3 tokens we just processed
                
            # Look for pattern: ▁ </*> (end of stressed word at word boundary)
            elif (i < len(tokens) - 1 and 
                  tokens[i] == self.WORD_BOUNDARY_TOKEN and 
                  tokens[i + 1] == self.STRESS_END_TOKEN):
                
                # Move </*/> before ▁
                fixed_tokens.extend([tokens[i + 1], tokens[i]])
                i += 2
                
            else:
                fixed_tokens.append(tokens[i])
                i += 1
                
        return fixed_tokens
    
    def create_training_pairs(self, text_with_stress: str, tokenizer) -> Tuple[List[str], List[str]]:
        """
        Create training pairs for fine-tuning.
        
        Args:
            text_with_stress: Text containing stress markers
            tokenizer: The text tokenizer
            
        Returns:
            Tuple of (input_tokens, target_tokens) for training
        """
        # Create input without stress tokens (for input text)
        text_without_stress = text_with_stress.replace(self.STRESS_START_TOKEN, '').replace(self.STRESS_END_TOKEN, '')
        
        # Create target with properly processed stress tokens
        target_tokens = tokenizer.tokenize(text_with_stress)
        target_tokens = self.fix_stress_tokens(target_tokens)
        
        # Input tokens (clean text)
        input_tokens = tokenizer.tokenize(text_without_stress)
        
        return input_tokens, target_tokens
    
    def prepare_training_data(self, texts_with_stress: List[str], tokenizer) -> List[Tuple[List[str], List[str]]]:
        """
        Prepare training data for fine-tuning the GPT module.
        
        Args:
            texts_with_stress: List of texts with stress markers
            tokenizer: The text tokenizer
            
        Returns:
            List of (input_tokens, target_tokens) pairs
        """
        training_pairs = []
        for text in texts_with_stress:
            input_tokens, target_tokens = self.create_training_pairs(text, tokenizer)
            training_pairs.append((input_tokens, target_tokens))
        return training_pairs
    
    def validate_tokens(self, tokens: List[str]) -> bool:
        """
        Validate that stress tokens don't break word boundaries.
        
        Args:
            tokens: List of tokens to validate
            
        Returns:
            True if tokens are properly formatted, False otherwise
        """
        for i in range(len(tokens) - 1):
            # Check if we have ▁ <*> pattern (bad)
            if (tokens[i] == self.WORD_BOUNDARY_TOKEN and 
                tokens[i + 1] == self.STRESS_START_TOKEN):
                return False
        return True
    
    def extract_stressed_words(self, text: str) -> List[str]:
        """
        Extract words that are marked with stress tokens.
        
        Args:
            text: Text containing stress markers
            
        Returns:
            List of stressed words
        """
        matches = self.stress_pattern.findall(text)
        return matches


# Example usage and testing
if __name__ == "__main__":
    import sys
    import os
    
    # Add the parent directory to path to import tokenizer
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from utils.front import TextTokenizer, TextNormalizer
    
    # Initialize
    processor = StressTokenProcessor()
    normalizer = TextNormalizer()
    normalizer.load()
    tokenizer = TextTokenizer('../../checkpoints/bpe_extended.model', normalizer)
    
    # Test text
    test_text = "and, if the Council is used, arrangements should be <*>made</*> for the attendance of the Secretary of the Treasury"
    
    print("=== Stress Token Processing Test ===")
    print(f"Original text: {test_text}")
    print()
    
    # Test original tokenization (problematic)
    original_tokens = tokenizer.tokenize(test_text)
    print("Original tokenization (problematic):")
    print(original_tokens)
    print(f"Valid: {processor.validate_tokens(original_tokens)}")
    print()
    
    # Test fixed tokenization
    fixed_tokens = processor.fix_stress_tokens(original_tokens)
    print("Fixed tokenization:")
    print(fixed_tokens)
    print(f"Valid: {processor.validate_tokens(fixed_tokens)}")
    print()
    
    # Test training pair creation
    input_tokens, target_tokens = processor.create_training_pairs(test_text, tokenizer)
    print("Training pair creation:")
    print(f"Input tokens (no stress):  {input_tokens}")
    print(f"Target tokens (with stress): {target_tokens}")
    print()
    
    # Extract stressed words
    stressed_words = processor.extract_stressed_words(test_text)
    print(f"Stressed words: {stressed_words}")