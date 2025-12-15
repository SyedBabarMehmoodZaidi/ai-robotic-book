"""
Validation reporter module.
This module generates comprehensive validation reports with quality metrics.
"""
import json
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, asdict
from config import logger


@dataclass
class ValidationReport:
    """Data class for validation report structure"""
    timestamp: str
    total_queries_executed: int
    successful_retrievals: int
    retrieval_success_rate: float
    average_similarity_score: float
    quality_threshold_met: bool
    quality_score: float
    metadata_accuracy: float
    metadata_accuracy_met: bool
    total_chunks_retrieved: int
    average_response_time: float
    validation_details: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    summary: Dict[str, Any]


def generate_validation_report(
    validation_results: List[Dict],
    metadata_accuracy: float,
    query_execution_times: Optional[List[float]] = None,
    total_queries: int = 0,
    successful_queries: int = 0
) -> ValidationReport:
    """
    Generates a comprehensive validation report with quality metrics

    Args:
        validation_results (List[Dict]): Results from chunk validation
        metadata_accuracy (float): Metadata accuracy percentage
        query_execution_times (List[float], optional): List of query execution times
        total_queries (int): Total number of queries executed
        successful_queries (int): Number of successful queries

    Returns:
        ValidationReport: Comprehensive validation report with metrics and statistics
    """
    from config import Config

    # Calculate metrics from validation results
    if validation_results and len(validation_results) > 1:  # Has overall stats at the end
        overall_stats = validation_results[-1].get('overall_stats', {})
        total_chunks = overall_stats.get('total_chunks', 0)
        relevant_chunks = overall_stats.get('relevant_chunks', 0)
        relevant_percentage = overall_stats.get('relevant_percentage', 0.0)
        quality_threshold_met = relevant_percentage >= (Config.QUALITY_THRESHOLD * 100)  # Convert to percentage
    else:
        total_chunks = 0
        relevant_chunks = 0
        relevant_percentage = 0.0
        quality_threshold_met = False

    # Calculate average similarity score if available
    if validation_results and len(validation_results) > 1:
        scores = []
        for result in validation_results[:-1]:  # Exclude overall stats
            if 'similarity_score' in result:
                scores.append(result['similarity_score'])
        average_similarity = sum(scores) / len(scores) if scores else 0.0
    else:
        average_similarity = 0.0

    # Calculate average response time if execution times provided
    average_response_time = 0.0
    if query_execution_times and len(query_execution_times) > 0:
        average_response_time = sum(query_execution_times) / len(query_execution_times)

    # Calculate retrieval success rate
    retrieval_success_rate = 0.0
    if total_queries > 0:
        retrieval_success_rate = (successful_queries / total_queries) * 100

    # Create performance metrics
    performance_metrics = {
        'total_queries': total_queries,
        'successful_queries': successful_queries,
        'failed_queries': total_queries - successful_queries,
        'success_rate': retrieval_success_rate,
        'total_chunks_retrieved': total_chunks,
        'relevant_chunks': relevant_chunks,
        'average_similarity_score': average_similarity,
        'average_response_time_ms': average_response_time,
        'metadata_accuracy': metadata_accuracy
    }

    # Create detailed validation summary
    metadata_accuracy_threshold = Config.METADATA_ACCURACY_THRESHOLD * 100  # Convert to percentage
    success_rate_threshold = Config.RETRIEVAL_SUCCESS_RATE_THRESHOLD * 100  # Convert to percentage

    summary = {
        'retrieval_quality_met': quality_threshold_met,
        'metadata_accuracy_met': metadata_accuracy >= metadata_accuracy_threshold,
        'overall_validation_passed': quality_threshold_met and metadata_accuracy >= metadata_accuracy_threshold,
        'recommendations': []
    }

    # Add recommendations based on results
    if not quality_threshold_met:
        summary['recommendations'].append(
            f"Quality threshold not met. Current: {relevant_percentage:.1f}%, Required: {Config.QUALITY_THRESHOLD * 100:.1f}%"
        )
    if metadata_accuracy < metadata_accuracy_threshold:
        summary['recommendations'].append(
            f"Metadata accuracy below threshold. Current: {metadata_accuracy:.1f}%, Required: {metadata_accuracy_threshold:.1f}%"
        )
    if retrieval_success_rate < success_rate_threshold:
        summary['recommendations'].append(
            f"Query success rate below threshold. Current: {retrieval_success_rate:.1f}%, Required: {success_rate_threshold:.1f}%"
        )

    # Create the validation report
    report = ValidationReport(
        timestamp=datetime.now().isoformat(),
        total_queries_executed=total_queries,
        successful_retrievals=successful_queries,
        retrieval_success_rate=retrieval_success_rate,
        average_similarity_score=average_similarity,
        quality_threshold_met=quality_threshold_met,
        quality_score=relevant_percentage,
        metadata_accuracy=metadata_accuracy,
        metadata_accuracy_met=metadata_accuracy >= metadata_accuracy_threshold,
        total_chunks_retrieved=total_chunks,
        average_response_time=average_response_time,
        validation_details=validation_results[:-1] if validation_results else [],  # Exclude overall stats
        performance_metrics=performance_metrics,
        summary=summary
    )

    logger.info(f"Generated validation report: {report.total_queries_executed} queries, "
                f"{report.retrieval_success_rate:.2f}% success rate, "
                f"{report.quality_score:.2f}% quality score")

    return report


def export_validation_report(report: ValidationReport, format: str = 'json',
                           filename: Optional[str] = None) -> str:
    """
    Export the validation report in the specified format

    Args:
        report (ValidationReport): The validation report to export
        format (str): Export format ('json', 'csv', or 'txt')
        filename (str, optional): Output filename (auto-generated if not provided)

    Returns:
        str: Path to the exported report file
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_report_{timestamp}.{format}"

    if format.lower() == 'json':
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(asdict(report), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to write JSON report to {filename}: {str(e)}")
            raise
    elif format.lower() == 'txt':
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("RAG Retrieval Validation Report\n")
                f.write("=" * 50 + "\n")
                f.write(f"Timestamp: {report.timestamp}\n")
                f.write(f"Total Queries Executed: {report.total_queries_executed}\n")
                f.write(f"Successful Retrievals: {report.successful_retrievals}\n")
                f.write(f"Retrieval Success Rate: {report.retrieval_success_rate:.2f}%\n")
                f.write(f"Quality Score: {report.quality_score:.2f}%\n")
                f.write(f"Quality Threshold Met: {'Yes' if report.quality_threshold_met else 'No'}\n")
                f.write(f"Metadata Accuracy: {report.metadata_accuracy:.2f}%\n")
                f.write(f"Metadata Accuracy Met: {'Yes' if report.metadata_accuracy_met else 'No'}\n")
                f.write(f"Average Response Time: {report.average_response_time:.2f}ms\n")
                f.write(f"Total Chunks Retrieved: {report.total_chunks_retrieved}\n")
                f.write(f"Average Similarity Score: {report.average_similarity_score:.2f}\n\n")

                f.write("Performance Metrics:\n")
                for key, value in report.performance_metrics.items():
                    f.write(f"  {key}: {value}\n")

                f.write(f"\nSummary:\n")
                for key, value in report.summary.items():
                    f.write(f"  {key}: {value}\n")
        except Exception as e:
            logger.error(f"Failed to write TXT report to {filename}: {str(e)}")
            raise
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Validation report exported to: {filename}")
    return filename


def print_validation_summary(report: ValidationReport):
    """
    Print a formatted summary of the validation report to console

    Args:
        report (ValidationReport): The validation report to display
    """
    print("\n" + "="*60)
    print("RAG RETRIEVAL VALIDATION REPORT")
    print("="*60)
    print(f"Timestamp: {report.timestamp}")
    print(f"Total Queries: {report.total_queries_executed}")
    print(f"Successful Retrievals: {report.successful_retrievals} ({report.retrieval_success_rate:.1f}%)")
    print(f"Average Similarity Score: {report.average_similarity_score:.3f}")
    print(f"Average Response Time: {report.average_response_time:.2f}ms")
    print(f"Total Chunks Retrieved: {report.total_chunks_retrieved}")
    print()

    print("QUALITY METRICS:")
    print(f"  Quality Score: {report.quality_score:.1f}% {'✅ PASS' if report.quality_threshold_met else '❌ FAIL'}")
    print(f"  Metadata Accuracy: {report.metadata_accuracy:.1f}% {'✅ PASS' if report.metadata_accuracy_met else '❌ FAIL'}")
    print(f"  Overall Validation: {'✅ PASS' if report.summary['overall_validation_passed'] else '❌ FAIL'}")
    print()

    if report.summary['recommendations']:
        print("RECOMMENDATIONS:")
        for rec in report.summary['recommendations']:
            print(f"  • {rec}")
    else:
        print("RECOMMENDATIONS: All validation criteria met! 🎉")

    print("="*60)


if __name__ == "__main__":
    # Example usage
    sample_validation_results = [
        {
            'chunk_index': 0,
            'chunk_id': 'chunk_1',
            'content_length': 150,
            'similarity_score': 0.92,
            'metadata_valid': True,
            'content_relevant': True,
            'relevance_score': 0.85,
            'validation_details': {'content_preview': 'Artificial intelligence is...'}
        },
        {
            'chunk_index': 1,
            'chunk_id': 'chunk_2',
            'content_length': 200,
            'similarity_score': 0.88,
            'metadata_valid': True,
            'content_relevant': True,
            'relevance_score': 0.78,
            'validation_details': {'content_preview': 'Machine learning algorithms...'}
        },
        {
            'overall_stats': {
                'total_chunks': 2,
                'relevant_chunks': 2,
                'relevant_percentage': 100.0,
                'valid_metadata_chunks': 2,
                'metadata_validity_percentage': 100.0
            }
        }
    ]

    sample_times = [125.5, 98.2, 142.7]

    report = generate_validation_report(
        validation_results=sample_validation_results,
        metadata_accuracy=100.0,
        query_execution_times=sample_times,
        total_queries=3,
        successful_queries=3
    )

    print_validation_summary(report)

    # Export the report
    export_path = export_validation_report(report, format='json')
    print(f"\nFull report exported to: {export_path}")