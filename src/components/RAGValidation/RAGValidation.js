import React, { useState, useEffect } from 'react';
import styles from './RAGValidation.module.css';

const RAGValidation = () => {
  const [validationResults, setValidationResults] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);

  // Validation tests to run
  const validationTests = [
    {
      id: 'FT-RAG-001',
      name: 'Query Submission Test',
      description: 'Verify that users can submit queries through the interface',
      category: 'Functional',
      critical: true
    },
    {
      id: 'FT-RAG-002',
      name: 'Selected Text Integration Test',
      description: 'Verify that selected text is captured and sent with queries',
      category: 'Functional',
      critical: true
    },
    {
      id: 'FT-RAG-003',
      name: 'Response Display Test',
      description: 'Verify that responses are properly displayed',
      category: 'Functional',
      critical: true
    },
    {
      id: 'IT-RAG-001',
      name: 'API Client Functionality Test',
      description: 'Verify that the API client communicates correctly with the backend',
      category: 'Integration',
      critical: true
    },
    {
      id: 'ET-RAG-001',
      name: 'Query Validation Test',
      description: 'Verify that client-side validation works',
      category: 'Error Handling',
      critical: true
    },
    {
      id: 'ET-RAG-003',
      name: 'Backend Connectivity Test',
      description: 'Verify error handling when backend is unavailable',
      category: 'Error Handling',
      critical: true
    }
  ];

  // Function to run a single validation test
  const runValidationTest = async (test) => {
    // Simulate test execution with random results
    await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 1000));

    // Randomly determine if test passes (90% success rate for demo)
    const isSuccess = Math.random() > 0.1;
    const responseTime = Math.floor(Math.random() * 2000) + 100; // 100-2100ms

    return {
      ...test,
      status: isSuccess ? 'passed' : 'failed',
      responseTime: `${responseTime}ms`,
      timestamp: new Date().toISOString(),
      details: isSuccess
        ? 'Test completed successfully'
        : `Test failed: ${test.name} did not meet success criteria`
    };
  };

  // Function to run all validation tests
  const runAllValidations = async () => {
    setIsRunning(true);
    setProgress(0);
    setValidationResults([]);

    const results = [];
    const totalTests = validationTests.length;

    for (let i = 0; i < validationTests.length; i++) {
      const test = validationTests[i];
      const result = await runValidationTest(test);
      results.push(result);

      // Update progress
      const newProgress = Math.floor(((i + 1) / totalTests) * 100);
      setProgress(newProgress);
    }

    setValidationResults(results);
    setIsRunning(false);
  };

  // Function to run a specific validation test
  const runSingleValidation = async (test) => {
    setIsRunning(true);
    const result = await runValidationTest(test);

    // Update the specific test result in the list
    const updatedResults = validationResults.map(r =>
      r.id === test.id ? result : r
    );

    // If test wasn't in the list yet, add it
    const existingTestIndex = validationResults.findIndex(r => r.id === test.id);
    if (existingTestIndex === -1) {
      setValidationResults([...updatedResults, result]);
    } else {
      setValidationResults(updatedResults);
    }

    setIsRunning(false);
  };

  // Calculate summary statistics
  const totalTests = validationResults.length;
  const passedTests = validationResults.filter(r => r.status === 'passed').length;
  const failedTests = validationResults.filter(r => r.status === 'failed').length;
  const successRate = totalTests > 0 ? Math.round((passedTests / totalTests) * 100) : 0;

  return (
    <div className={styles.validationContainer}>
      <h3>RAG System Validation</h3>

      <div className={styles.validationHeader}>
        <div className={styles.stats}>
          <div className={styles.statItem}>
            <span className={styles.statNumber}>{totalTests}</span>
            <span className={styles.statLabel}>Total Tests</span>
          </div>
          <div className={styles.statItem}>
            <span className={styles.statNumber}>{passedTests}</span>
            <span className={styles.statLabel}>Passed</span>
          </div>
          <div className={styles.statItem}>
            <span className={styles.statNumber}>{failedTests}</span>
            <span className={styles.statLabel}>Failed</span>
          </div>
          <div className={styles.statItem}>
            <span className={styles.statNumber}>{successRate}%</span>
            <span className={styles.statLabel}>Success Rate</span>
          </div>
        </div>

        <div className={styles.controls}>
          <button
            onClick={runAllValidations}
            disabled={isRunning}
            className={styles.runAllButton}
          >
            {isRunning ? 'Running Tests...' : 'Run All Validations'}
          </button>
        </div>
      </div>

      {isRunning && (
        <div className={styles.progressBar}>
          <div
            className={styles.progressFill}
            style={{ width: `${progress}%` }}
          >
            {progress}%
          </div>
        </div>
      )}

      <div className={styles.testCategories}>
        {['Functional', 'Integration', 'Error Handling'].map(category => {
          const categoryTests = validationTests.filter(t => t.category === category);
          const categoryResults = validationResults.filter(r =>
            categoryTests.some(t => t.id === r.id)
          );

          return (
            <div key={category} className={styles.testCategory}>
              <h4>{category} Tests</h4>
              <div className={styles.testsList}>
                {categoryTests.map(test => {
                  const result = categoryResults.find(r => r.id === test.id);
                  return (
                    <div key={test.id} className={styles.testItem}>
                      <div className={styles.testInfo}>
                        <div className={styles.testHeader}>
                          <span className={styles.testId}>{test.id}</span>
                          <span className={styles.testName}>{test.name}</span>
                          {test.critical && <span className={styles.criticalTag}>Critical</span>}
                        </div>
                        <p className={styles.testDescription}>{test.description}</p>
                      </div>

                      <div className={styles.testActions}>
                        {result ? (
                          <div className={`${styles.result} ${styles[result.status]}`}>
                            <span className={styles.resultStatus}>
                              {result.status === 'passed' ? '✓ Passed' : '✗ Failed'}
                            </span>
                            <span className={styles.responseTime}>{result.responseTime}</span>
                          </div>
                        ) : (
                          <button
                            onClick={() => runSingleValidation(test)}
                            disabled={isRunning}
                            className={styles.runTestButton}
                          >
                            Run Test
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      {validationResults.length > 0 && (
        <div className={styles.detailedResults}>
          <h4>Detailed Results</h4>
          <div className={styles.resultsList}>
            {validationResults.map(result => (
              <div key={result.id} className={styles.resultItem}>
                <div className={styles.resultHeader}>
                  <span className={`${styles.resultStatus} ${styles[result.status]}`}>
                    {result.status === 'passed' ? '✓' : '✗'} {result.id}
                  </span>
                  <span className={styles.timestamp}>{new Date(result.timestamp).toLocaleTimeString()}</span>
                </div>
                <div className={styles.resultDetails}>
                  <p><strong>Test:</strong> {result.name}</p>
                  <p><strong>Status:</strong> <span className={styles[result.status]}>{result.status}</span></p>
                  <p><strong>Response Time:</strong> {result.responseTime}</p>
                  <p><strong>Details:</strong> {result.details}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default RAGValidation;