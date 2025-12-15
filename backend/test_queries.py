"""
Test query suite for validating retrieval quality.
This file contains predefined test queries with expected results for validation.
"""
from typing import List, Dict, NamedTuple


class TestQuery(NamedTuple):
    """Represents a test query with expected results"""
    query: str
    expected_keywords: List[str]
    category: str
    description: str


class TestQuerySuite:
    """A suite of test queries for validating retrieval quality"""

    def __init__(self):
        self.queries = [
            TestQuery(
                query="What is artificial intelligence?",
                expected_keywords=["artificial intelligence", "AI", "machine learning", "intelligent", "systems"],
                category="definition",
                description="Basic definition of AI"
            ),
            TestQuery(
                query="Explain neural networks and deep learning",
                expected_keywords=["neural network", "deep learning", "layers", "weights", "training"],
                category="technical",
                description="Technical explanation of neural networks"
            ),
            TestQuery(
                query="What are the applications of robotics in industry?",
                expected_keywords=["robotics", "industrial", "automation", "manufacturing", "assembly"],
                category="applications",
                description="Industrial applications of robotics"
            ),
            TestQuery(
                query="How does computer vision work?",
                expected_keywords=["computer vision", "image processing", "object detection", "recognition", "pixels"],
                category="technical",
                description="Explanation of computer vision"
            ),
            TestQuery(
                query="What is natural language processing?",
                expected_keywords=["natural language processing", "NLP", "text", "language", "understanding"],
                category="definition",
                description="Definition of NLP"
            ),
            TestQuery(
                query="Explain reinforcement learning algorithms",
                expected_keywords=["reinforcement learning", "RL", "rewards", "agents", "policy"],
                category="technical",
                description="Technical explanation of RL"
            ),
            TestQuery(
                query="What are the ethical considerations in AI?",
                expected_keywords=["ethics", "AI", "bias", "fairness", "transparency"],
                category="ethics",
                description="Ethical considerations in AI"
            ),
            TestQuery(
                query="How is machine learning used in healthcare?",
                expected_keywords=["machine learning", "healthcare", "medical", "diagnosis", "treatment"],
                category="applications",
                description="Healthcare applications of ML"
            ),
            TestQuery(
                query="What is the difference between supervised and unsupervised learning?",
                expected_keywords=["supervised", "unsupervised", "learning", "labeled", "training"],
                category="comparison",
                description="Comparison of learning types"
            ),
            TestQuery(
                query="Explain the concept of overfitting in machine learning",
                expected_keywords=["overfitting", "machine learning", "training", "generalization", "complexity"],
                category="technical",
                description="Explanation of overfitting"
            )
        ]

    def get_queries_by_category(self, category: str) -> List[TestQuery]:
        """Get test queries filtered by category"""
        return [q for q in self.queries if q.category == category]

    def get_all_queries(self) -> List[TestQuery]:
        """Get all test queries"""
        return self.queries

    def get_expected_keywords_for_query(self, query_text: str) -> List[str]:
        """Get expected keywords for a specific query text"""
        for q in self.queries:
            if q.query.lower() == query_text.lower():
                return q.expected_keywords
        return []

    def evaluate_retrieval_quality(self, query: str, retrieved_content: str) -> Dict:
        """
        Evaluate the quality of retrieved content for a specific query
        """
        test_query = None
        for tq in self.queries:
            if tq.query.lower() == query.lower():
                test_query = tq
                break

        if not test_query:
            return {
                'query': query,
                'found_keywords': [],
                'missing_keywords': [],
                'keyword_coverage': 0.0,
                'relevance_score': 0.0
            }

        # Convert content to lowercase for matching
        content_lower = retrieved_content.lower()

        # Find expected keywords in the content
        found_keywords = []
        for keyword in test_query.expected_keywords:
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)

        # Calculate metrics
        total_expected = len(test_query.expected_keywords)
        found_count = len(found_keywords)
        keyword_coverage = found_count / total_expected if total_expected > 0 else 0.0

        # Simple relevance score based on keyword coverage
        relevance_score = keyword_coverage

        return {
            'query': query,
            'expected_keywords': test_query.expected_keywords,
            'found_keywords': found_keywords,
            'missing_keywords': [k for k in test_query.expected_keywords if k not in found_keywords],
            'keyword_coverage': keyword_coverage,
            'relevance_score': relevance_score,
            'category': test_query.category
        }


# Create a global instance of the test suite
test_suite = TestQuerySuite()


if __name__ == "__main__":
    # Example usage
    print("Test Query Suite:")
    print(f"Total queries: {len(test_suite.get_all_queries())}")

    for i, query in enumerate(test_suite.get_all_queries(), 1):
        print(f"{i}. {query.query}")
        print(f"   Category: {query.category}")
        print(f"   Expected keywords: {query.expected_keywords}")
        print(f"   Description: {query.description}")
        print()

    # Example evaluation
    sample_content = "Artificial intelligence is a wonderful field that involves creating intelligent systems using machine learning algorithms."
    evaluation = test_suite.evaluate_retrieval_quality("What is artificial intelligence?", sample_content)
    print("Sample evaluation:")
    print(f"Query: {evaluation['query']}")
    print(f"Found keywords: {evaluation['found_keywords']}")
    print(f"Keyword coverage: {evaluation['keyword_coverage']:.2f}")
    print(f"Relevance score: {evaluation['relevance_score']:.2f}")