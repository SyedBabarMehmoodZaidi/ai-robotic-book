"""
Main execution script for RAG Retrieval Validation Pipeline.
This script runs the complete validation pipeline from query input to final report generation.
"""
import sys
import os
import argparse
import json
from typing import List, Optional
import logging

# Add the backend directory to the path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from integration import run_complete_validation_pipeline
from test_queries import test_suite
from config import Config, validate_environment_variables, setup_logging
from validation_reporter import export_validation_report


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RAG Retrieval Validation Pipeline')
    parser.add_argument('--queries', '-q', nargs='+', help='List of queries to validate')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to retrieve (default: 5)')
    parser.add_argument('--test-suite', action='store_true', help='Use test suite queries instead of custom queries')
    parser.add_argument('--output-format', choices=['json', 'txt'], default='json', help='Output format for validation report (default: json)')
    parser.add_argument('--output-file', help='Output file path for validation report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    return parser.parse_args()


def load_queries_from_file(file_path: str) -> List[str]:
    """Load queries from a text file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Query file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]

    return queries


def main():
    """Main execution function"""
    print("🤖 Starting RAG Retrieval Validation Pipeline...")

    # Parse command line arguments
    args = parse_arguments()

    # Set up logging
    logger = setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Validate environment variables
        print("🔍 Validating environment variables...")
        validate_environment_variables()
        print("✅ Environment variables validated successfully")

        # Determine queries to use
        if args.test_suite:
            print("📋 Using test suite queries...")
            queries = [q.query for q in test_suite.get_all_queries()]
            print(f"   Loaded {len(queries)} test suite queries")
        elif args.queries:
            queries = args.queries
            print(f"📋 Using provided queries: {queries}")
        else:
            # Default to a few sample queries if none provided
            print("📋 Using sample queries...")
            queries = [
                "What is artificial intelligence?",
                "Explain neural networks and deep learning",
                "How does machine learning work?"
            ]
            print(f"   Using default sample queries: {queries}")

        # Run the complete validation pipeline
        print(f"\n🚀 Running validation pipeline for {len(queries)} queries...")
        print(f"   Top-K: {args.top_k}")

        results = run_complete_validation_pipeline(queries, top_k=args.top_k)

        # Export validation report
        print(f"\n📊 Generating validation report...")
        report = results['pipeline_results']['report']

        if args.output_file:
            output_path = export_validation_report(
                report,
                format=args.output_format,
                filename=args.output_file
            )
        else:
            # Generate a default filename with timestamp
            output_path = export_validation_report(report, format=args.output_format)

        print(f"✅ Validation report exported to: {output_path}")

        # Print summary statistics
        print(f"\n📈 Pipeline Summary:")
        print(f"   Total queries processed: {report.total_queries_executed}")
        print(f"   Successful retrievals: {report.successful_retrievals}")
        print(f"   Success rate: {report.retrieval_success_rate:.2f}%")
        print(f"   Quality score: {report.quality_score:.2f}%")
        print(f"   Metadata accuracy: {report.metadata_accuracy:.2f}%")
        print(f"   Average response time: {report.average_response_time:.2f}ms")
        print(f"   Total chunks retrieved: {report.total_chunks_retrieved}")

        # Check if validation criteria were met
        quality_passed = report.quality_threshold_met
        metadata_passed = report.metadata_accuracy_met
        overall_passed = report.summary['overall_validation_passed']

        print(f"\n✅ Validation Results:")
        print(f"   Quality threshold (90%) met: {'YES' if quality_passed else 'NO'}")
        print(f"   Metadata accuracy (100%) met: {'YES' if metadata_passed else 'NO'}")
        print(f"   Overall validation passed: {'YES' if overall_passed else 'NO'}")

        if not overall_passed:
            print(f"\n⚠️  Recommendations:")
            for rec in report.summary['recommendations']:
                print(f"   • {rec}")

        print(f"\n🎉 RAG Retrieval Validation Pipeline completed successfully!")

        # Return success code based on validation results
        return 0 if overall_passed else 1

    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        return 130  # Standard exit code for Ctrl+C
    except FileNotFoundError as e:
        print(f"\n❌ File not found error: {e}")
        return 1
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_interactive_mode():
    """Run the validation pipeline in interactive mode"""
    print("🤖 RAG Retrieval Validation Pipeline - Interactive Mode")
    print("=" * 60)

    while True:
        try:
            print("\nOptions:")
            print("1. Run with sample queries")
            print("2. Enter custom queries")
            print("3. Load queries from file")
            print("4. Run test suite queries")
            print("5. Exit")

            choice = input("\nSelect an option (1-5): ").strip()

            if choice == '1':
                queries = [
                    "What is artificial intelligence?",
                    "Explain neural networks",
                    "How does machine learning work?"
                ]
                print(f"\nUsing sample queries: {queries}")
            elif choice == '2':
                query_input = input("Enter queries separated by semicolons: ").strip()
                queries = [q.strip() for q in query_input.split(';') if q.strip()]
                if not queries:
                    print("❌ No queries entered")
                    continue
                print(f"\nUsing custom queries: {queries}")
            elif choice == '3':
                file_path = input("Enter path to queries file: ").strip()
                queries = load_queries_from_file(file_path)
                print(f"\nLoaded {len(queries)} queries from file: {file_path}")
            elif choice == '4':
                queries = [q.query for q in test_suite.get_all_queries()]
                print(f"\nUsing test suite queries: {len(queries)} queries loaded")
            elif choice == '5':
                print("\n👋 Exiting interactive mode...")
                break
            else:
                print("❌ Invalid option. Please select 1-5.")
                continue

            # Get top-k value
            try:
                top_k_input = input(f"Enter top-k value (default 5): ").strip()
                top_k = int(top_k_input) if top_k_input else 5
            except ValueError:
                top_k = 5
                print("Using default top-k value: 5")

            # Run validation
            print(f"\n🚀 Running validation for {len(queries)} queries with top-k={top_k}...")
            results = run_complete_validation_pipeline(queries, top_k=top_k)

            # Export report
            report = results['pipeline_results']['report']
            output_path = export_validation_report(report, format='txt')
            print(f"✅ Report exported to: {output_path}")

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    print("👋 Interactive mode ended.")


if __name__ == "__main__":
    # Check if running in interactive mode
    if len(sys.argv) == 1:
        # No arguments provided, run interactive mode
        run_interactive_mode()
    else:
        # Arguments provided, run in batch mode
        exit_code = main()
        sys.exit(exit_code)
