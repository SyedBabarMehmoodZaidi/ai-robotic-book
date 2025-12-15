"""
Hallucination detection utilities for the RAG agent.
This module provides functions to detect potential hallucinations in agent responses.
"""
from typing import List, Dict, Tuple
import re
from ..models import RetrievedContext


class HallucinationDetector:
    """
    A class to detect potential hallucinations in AI-generated responses.
    """

    @staticmethod
    def detect_hallucinations(response_text: str, contexts: List[RetrievedContext]) -> Tuple[bool, Dict[str, any]]:
        """
        Detect potential hallucinations in the response text based on provided contexts.

        Args:
            response_text (str): The AI-generated response text
            contexts (List[RetrievedContext]): The contexts used to generate the response

        Returns:
            Tuple[bool, Dict]: (is_hallucination_detected, details about the detection)
        """
        details = {
            'confidence': 0.0,
            'issues_found': [],
            'supporting_evidence': [],
            'confidence_sources': []
        }

        # Check 1: Verify that claims in the response are supported by contexts
        unsupported_claims = HallucinationDetector._find_unsupported_claims(response_text, contexts)
        if unsupported_claims:
            details['issues_found'].extend(unsupported_claims)
            details['confidence'] -= len(unsupported_claims) * 0.2  # Reduce confidence for each unsupported claim

        # Check 2: Look for certainty markers when information isn't clearly supported
        certainty_without_support = HallucinationDetector._check_certainty_without_support(response_text, contexts)
        if certainty_without_support:
            details['issues_found'].append("High certainty statements without clear support from context")
            details['confidence'] -= 0.3

        # Check 3: Look for statistical or factual claims that can't be verified from context
        unverifiable_facts = HallucinationDetector._find_unverifiable_facts(response_text, contexts)
        if unverifiable_facts:
            details['issues_found'].extend(unverifiable_facts)
            details['confidence'] -= len(unverifiable_facts) * 0.15

        # Calculate final confidence (ensure it's between 0 and 1)
        details['confidence'] = max(0.0, min(1.0, 1.0 + details['confidence']))

        # Determine if hallucination is detected based on issues found and confidence
        is_hallucination = len(details['issues_found']) > 0 and details['confidence'] < 0.5

        return is_hallucination, details

    @staticmethod
    def _find_unsupported_claims(response_text: str, contexts: List[RetrievedContext]) -> List[str]:
        """
        Find claims in the response that are not supported by the contexts.

        Args:
            response_text (str): The response text to analyze
            contexts (List[RetrievedContext]): The contexts to check against

        Returns:
            List[str]: List of unsupported claims
        """
        unsupported_claims = []

        # This is a simplified implementation
        # In a production system, this would use more sophisticated NLP techniques
        response_sentences = re.split(r'[.!?]+', response_text)
        context_text = " ".join([ctx.content for ctx in contexts]).lower()

        for sentence in response_sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Only check substantial sentences
                sentence_lower = sentence.lower()
                # Check if sentence content appears in contexts
                if not any(word in context_text for word in sentence_lower.split() if len(word) > 3):
                    unsupported_claims.append(sentence.strip())

        return unsupported_claims

    @staticmethod
    def _check_certainty_without_support(response_text: str, contexts: List[RetrievedContext]) -> bool:
        """
        Check if the response contains high-certainty language without clear support from contexts.

        Args:
            response_text (str): The response text to analyze
            contexts (List[RetrievedContext]): The contexts to check against

        Returns:
            bool: True if certainty without support is detected
        """
        # Look for certainty markers in the response
        certainty_patterns = [
            r'\balways\b', r'\bnever\b', r'\bevery\b', r'\ball\b',
            r'\bdefinitely\b', r'\bsurely\b', r'\bcertainly\b', r'\babsolutely\b',
            r'\b100%', r'\bno doubt', r'\bwithout doubt'
        ]

        has_certainty = any(re.search(pattern, response_text, re.IGNORECASE) for pattern in certainty_patterns)

        if has_certainty:
            # Check if the certainty statements are supported by contexts
            context_text = " ".join([ctx.content for ctx in contexts]).lower()
            response_lower = response_text.lower()

            # If there's certainty but limited overlap with context, it might be hallucination
            common_words = set(response_lower.split()) & set(context_text.split())
            overlap_ratio = len(common_words) / max(len(set(response_lower.split())), 1)

            return overlap_ratio < 0.3  # If less than 30% of words overlap, flag as potential issue

        return False

    @staticmethod
    def _find_unverifiable_facts(response_text: str, contexts: List[RetrievedContext]) -> List[str]:
        """
        Find specific facts in the response that cannot be verified from contexts.

        Args:
            response_text (str): The response text to analyze
            contexts (List[RetrievedContext]): The contexts to check against

        Returns:
            List[str]: List of potentially unverifiable facts
        """
        unverifiable_facts = []

        # Look for patterns that indicate specific facts that should be in contexts
        # e.g., dates, statistics, specific numbers, named entities
        date_pattern = r'\b\d{4}\b'  # Potential years
        number_pattern = r'\b\d{3,}\b'  # Numbers with 3+ digits
        stat_pattern = r'\b\d+\.?\d*\s*(%|percent|of|times|average|average|mean|median)\b'

        dates = re.findall(date_pattern, response_text)
        numbers = re.findall(number_pattern, response_text)
        stats = re.findall(stat_pattern, response_text, re.IGNORECASE)

        # Check if these specific facts appear in contexts
        context_text = " ".join([ctx.content for ctx in contexts]).lower()

        for date in dates:
            if date not in context_text:
                unverifiable_facts.append(f"Date/Year '{date}' not found in context")

        for number in numbers:
            if len(number) > 2 and number not in context_text:  # Focus on larger numbers
                unverifiable_facts.append(f"Number '{number}' not found in context")

        for stat in stats:
            if stat.lower() not in context_text.lower():
                unverifiable_facts.append(f"Statistical claim '{stat}' not found in context")

        return unverifiable_facts


# Create a singleton instance of the hallucination detector
hallucination_detector = HallucinationDetector()