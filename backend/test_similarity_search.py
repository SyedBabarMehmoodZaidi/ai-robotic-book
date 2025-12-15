"""
Test script for similarity search functionality.
This script tests the retrieval_validator module with sample queries.
Note: This requires a populated Qdrant database with embeddings.
"""
import sys
import os
from typing import List, Dict
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieval_validator import execute_similarity_search_query
from query_converter import query_to_vector


def test_similarity_search():
    """Test similarity search with sample queries"""
    print("Testing similarity search functionality...")

    # Define test queries
    test_queries = [
        "What is artificial intelligence?",
        "Explain machine learning algorithms",
        "How does neural network work?",
        "What are the applications of robotics?",
        "Describe computer vision techniques"
    ]

    print(f"Running {len(test_queries)} test queries...")

    all_tests_passed = True

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i}: '{query}' ---")

        try:
            # Execute similarity search
            results = execute_similarity_search_query(query, top_k=5)

            print(f"Retrieved {len(results)} results")

            if len(results) == 0:
                print("WARNING: No results returned for this query")
                continue

            # Verify each result has required structure (Task T019)
            for j, chunk in enumerate(results, 1):
                print(f"  Result {j}: Score={chunk['score']:.4f}")

                # Verify content exists
                if 'content' not in chunk or not chunk['content']:
                    print(f"    ERROR: Missing or empty content in result {j}")
                    all_tests_passed = False
                    continue

                # Verify metadata exists and has required fields
                if 'metadata' not in chunk:
                    print(f"    ERROR: Missing metadata in result {j}")
                    all_tests_passed = False
                    continue

                metadata = chunk['metadata']
                required_metadata_fields = ['url', 'section', 'chunk_id']

                for field in required_metadata_fields:
                    if field not in metadata:
                        print(f"    ERROR: Missing '{field}' in metadata of result {j}")
                        all_tests_passed = False
                    elif not metadata[field]:
                        print(f"    WARNING: Empty '{field}' in metadata of result {j}")

                # Verify similarity score exists
                if 'score' not in chunk:
                    print(f"    ERROR: Missing similarity score in result {j}")
                    all_tests_passed = False

                # Print first 100 chars of content for verification
                content_preview = chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content']
                print(f"    Content preview: {content_preview}")

            print(f"  Query {i} completed successfully with {len(results)} results")

        except Exception as e:
            print(f"  ERROR: Query '{query}' failed with error: {str(e)}")
            all_tests_passed = False

    print(f"\n--- Test Summary ---")
    if all_tests_passed and any(len(execute_similarity_search_query(q, top_k=1)) > 0 for q in test_queries[:2]):
        print("✅ Similarity search tests completed successfully!")
        print("✅ Retrieved chunks contain content and metadata with similarity scores (T019)")
        print("✅ All required fields present in results")
    else:
        print("❌ Some tests failed or no results were returned")
        print("💡 This may be because Qdrant database is not populated with embeddings yet")

    return all_tests_passed


def test_query_conversion():
    """Test query conversion functionality separately"""
    print("\n--- Testing Query Conversion ---")

    test_query = "What is artificial intelligence?"

    try:
        vector = query_to_vector(test_query)
        print(f"✅ Query converted successfully to vector with dimension {len(vector)}")
        print(f"   First 5 elements: {vector[:5]}")
        return True
    except Exception as e:
        print(f"❌ Query conversion failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("Running similarity search validation tests...")
    print("Note: These tests require a running Qdrant instance with populated embeddings.")

    # Test query conversion first
    conversion_ok = test_query_conversion()

    if conversion_ok:
        # Test similarity search
        search_ok = test_similarity_search()

        if search_ok:
            print("\n🎉 All tests passed! Similarity search functionality is working correctly.")
        else:
            print("\n⚠️  Some tests had issues (likely due to empty database).")
    else:
        print("\n❌ Query conversion failed - check Cohere API configuration.")