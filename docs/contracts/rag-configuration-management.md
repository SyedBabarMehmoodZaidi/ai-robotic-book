# RAG Configuration Management

This document outlines the configuration management system for the Retrieval-Augmented Generation (RAG) system in the Physical AI & Humanoid Robotics book.

## Overview

The RAG system implements a flexible configuration management system that allows for different settings in development, testing, and production environments. This document covers the configuration structure, management practices, and deployment considerations.

## Configuration Structure

### 1. Configuration Parameters

#### Core API Configuration
```javascript
const RAGConfig = {
  // Backend API URL - defaults to local development
  BACKEND_API_URL: process.env.BACKEND_API_URL || 'http://localhost:8000',

  // Development mode flag
  DEVELOPMENT_MODE: process.env.NODE_ENV !== 'production',

  // Request timeout in milliseconds
  REQUEST_TIMEOUT: 30000, // 30 seconds

  // Retry settings
  MAX_RETRIES: 2,
  RETRY_DELAY: 1000, // 1 second

  // Validation settings
  MIN_QUERY_LENGTH: 3,
  MAX_QUERY_LENGTH: 2000,
  MIN_SELECTED_TEXT_LENGTH: 10,
  MAX_SELECTED_TEXT_LENGTH: 5000
};
```

#### Parameter Descriptions
- **BACKEND_API_URL**: Base URL for the RAG backend service
- **DEVELOPMENT_MODE**: Flag indicating development vs production mode
- **REQUEST_TIMEOUT**: Maximum time to wait for API responses
- **MAX_RETRIES**: Number of retry attempts for failed requests
- **RETRY_DELAY**: Delay between retry attempts
- **MIN_QUERY_LENGTH**: Minimum length for query text validation
- **MAX_QUERY_LENGTH**: Maximum length for query text validation
- **MIN_SELECTED_TEXT_LENGTH**: Minimum length for selected text validation
- **MAX_SELECTED_TEXT_LENGTH**: Maximum length for selected text validation

### 2. Environment-Specific Configuration

#### Development Configuration
```javascript
{
  BACKEND_API_URL: 'http://localhost:8000',
  DEVELOPMENT_MODE: true,
  REQUEST_TIMEOUT: 60000,        // Longer timeout for development
  MAX_RETRIES: 3,               // More retries during development
  LOG_LEVEL: 'debug'            // Detailed logging
}
```

#### Production Configuration
```javascript
{
  BACKEND_API_URL: 'https://rag-api.example.com',
  DEVELOPMENT_MODE: false,
  REQUEST_TIMEOUT: 30000,        // Standard timeout
  MAX_RETRIES: 2,               // Standard retries
  LOG_LEVEL: 'error'            // Minimal logging
}
```

#### Testing Configuration
```javascript
{
  BACKEND_API_URL: 'http://test-rag-api.example.com',
  DEVELOPMENT_MODE: true,
  REQUEST_TIMEOUT: 10000,        // Short timeout for tests
  MAX_RETRIES: 0,               // No retries in tests
  LOG_LEVEL: 'silent'           // No logging in tests
}
```

## Configuration Management Implementation

### 1. Frontend Configuration Loading

#### Global Configuration Object
The configuration is loaded globally and made available to all components:

```javascript
// Configuration for RAG frontend integration
const RAGConfig = {
  // Backend API URL - defaults to local development
  BACKEND_API_URL: process.env.BACKEND_API_URL || 'http://localhost:8000',

  // Development mode flag
  DEVELOPMENT_MODE: process.env.NODE_ENV !== 'production',

  // Request timeout in milliseconds
  REQUEST_TIMEOUT: 30000, // 30 seconds

  // Retry settings
  MAX_RETRIES: 2,
  RETRY_DELAY: 1000, // 1 second

  // Validation settings
  MIN_QUERY_LENGTH: 3,
  MAX_QUERY_LENGTH: 2000,
  MIN_SELECTED_TEXT_LENGTH: 10,
  MAX_SELECTED_TEXT_LENGTH: 5000
};

// Make config available globally
window.RAGConfig = RAGConfig;

export default RAGConfig;
```

#### API Client Configuration Usage
```javascript
class RAGApiClient {
  constructor(baseURL = null) {
    // Use config if available, otherwise default
    this.baseURL = baseURL || (typeof window.RAGConfig !== 'undefined' ? window.RAGConfig.BACKEND_API_URL : 'http://localhost:8000');
    this.config = typeof window.RAGConfig !== 'undefined' ? window.RAGConfig : {
      REQUEST_TIMEOUT: 30000,
      MAX_RETRIES: 2,
      RETRY_DELAY: 1000
    };
  }
}
```

### 2. Configuration Validation

#### Runtime Validation
```javascript
// Validate configuration at runtime
function validateConfig(config) {
  const errors = [];

  if (typeof config.BACKEND_API_URL !== 'string' || !config.BACKEND_API_URL) {
    errors.push('BACKEND_API_URL must be a valid string');
  }

  if (typeof config.REQUEST_TIMEOUT !== 'number' || config.REQUEST_TIMEOUT <= 0) {
    errors.push('REQUEST_TIMEOUT must be a positive number');
  }

  if (typeof config.MAX_RETRIES !== 'number' || config.MAX_RETRIES < 0) {
    errors.push('MAX_RETRIES must be a non-negative number');
  }

  if (typeof config.MIN_QUERY_LENGTH !== 'number' || config.MIN_QUERY_LENGTH <= 0) {
    errors.push('MIN_QUERY_LENGTH must be a positive number');
  }

  if (typeof config.MAX_QUERY_LENGTH !== 'number' || config.MAX_QUERY_LENGTH <= 0) {
    errors.push('MAX_QUERY_LENGTH must be a positive number');
  }

  if (config.MIN_QUERY_LENGTH > config.MAX_QUERY_LENGTH) {
    errors.push('MIN_QUERY_LENGTH cannot exceed MAX_QUERY_LENGTH');
  }

  return {
    isValid: errors.length === 0,
    errors
  };
}
```

## Configuration Management Patterns

### 1. Environment Variables

#### Frontend Environment Variables
- **BACKEND_API_URL**: API endpoint URL
- **NODE_ENV**: Environment indicator (development/production)
- **REACT_APP_**: Prefix for React-specific variables (if using Create React App)

#### Configuration Loading Order
1. Environment variables
2. Configuration files
3. Default values

### 2. Configuration Files

#### Static Configuration File
- **File Location**: `static/js/config.js`
- **Global Access**: Available via `window.RAGConfig`
- **Format**: JavaScript module with JSON-compatible structure

#### Dynamic Configuration
- **Runtime Updates**: Configuration can be updated at runtime
- **Component-Specific**: Components can override global settings
- **User Preferences**: User-specific configuration options

## Deployment Configuration

### 1. Build-Time Configuration

#### Environment-Specific Builds
```bash
# Development build
npm run build:dev

# Production build
npm run build:prod

# Staging build
npm run build:staging
```

#### Build Configuration Variables
- **API Endpoints**: Different endpoints for each environment
- **Feature Flags**: Enable/disable features per environment
- **Analytics Keys**: Different keys for each environment

### 2. Runtime Configuration

#### Dynamic Configuration Loading
- **AJAX Loading**: Load configuration from API endpoint
- **Feature Toggles**: Enable/disable features dynamically
- **A/B Testing**: Different configurations for testing

#### Configuration Caching
- **Browser Cache**: Cache configuration in browser storage
- **In-Memory Cache**: Cache configuration in memory
- **Cache Expiration**: Time-based cache invalidation

## Configuration Security

### 1. Secure Configuration Handling

#### Environment Variable Security
- **No Secrets in Frontend**: Never store secrets in frontend configuration
- **Public Configuration Only**: Only expose public settings
- **Validation**: Validate configuration values

#### Configuration Encryption
- **At Rest**: Encrypt sensitive configuration files
- **In Transit**: Secure configuration transmission
- **Runtime Protection**: Protect configuration in memory

### 2. Configuration Access Control

#### Role-Based Configuration
- **User Roles**: Different configurations for different user roles
- **Feature Access**: Configuration-based feature access
- **Permission Levels**: Configuration permissions

#### Configuration Auditing
- **Access Logs**: Log configuration access
- **Change Tracking**: Track configuration changes
- **Audit Trails**: Maintain configuration audit trails

## Configuration Testing

### 1. Unit Testing Configuration

#### Configuration Validation Tests
```javascript
// Test configuration validation
describe('RAG Configuration', () => {
  test('should validate valid configuration', () => {
    const config = {
      BACKEND_API_URL: 'http://localhost:8000',
      REQUEST_TIMEOUT: 30000,
      MAX_RETRIES: 2,
      MIN_QUERY_LENGTH: 3,
      MAX_QUERY_LENGTH: 2000
    };

    const result = validateConfig(config);
    expect(result.isValid).toBe(true);
  });

  test('should reject invalid configuration', () => {
    const config = {
      BACKEND_API_URL: '',
      REQUEST_TIMEOUT: -1,
      MAX_RETRIES: -1
    };

    const result = validateConfig(config);
    expect(result.isValid).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
  });
});
```

#### Environment Testing
- **Development Environment**: Test development-specific settings
- **Production Environment**: Test production-specific settings
- **Cross-Environment**: Test configuration consistency

### 2. Integration Testing

#### API Client Configuration Tests
- **Timeout Testing**: Test timeout functionality
- **Retry Testing**: Test retry mechanisms
- **URL Testing**: Test different API endpoints

#### Component Configuration Tests
- **Component Behavior**: Test component behavior with different configs
- **State Management**: Test configuration-based state changes
- **User Interface**: Test UI changes based on configuration

## Configuration Monitoring

### 1. Configuration Validation

#### Runtime Validation
- **Continuous Validation**: Validate configuration during runtime
- **Error Reporting**: Report configuration errors
- **Fallback Handling**: Handle configuration failures

#### Configuration Drift Detection
- **Baseline Comparison**: Compare current config to baseline
- **Anomaly Detection**: Detect unexpected configuration changes
- **Alerting**: Alert on configuration issues

### 2. Configuration Analytics

#### Usage Analytics
- **Feature Usage**: Track feature usage based on configuration
- **Performance Metrics**: Monitor performance by configuration
- **User Behavior**: Analyze user behavior with different configs

#### Configuration Effectiveness
- **A/B Testing Results**: Measure configuration effectiveness
- **Performance Impact**: Assess performance impact of configs
- **User Satisfaction**: Track user satisfaction by config

## Configuration Best Practices

### 1. Development Best Practices

#### Configuration Management
- **Consistent Naming**: Use consistent configuration parameter names
- **Documentation**: Document all configuration parameters
- **Validation**: Implement configuration validation
- **Defaults**: Provide sensible default values

#### Version Control
- **Configuration Files**: Include configuration files in version control
- **Environment Variables**: Document required environment variables
- **Secrets Management**: Use separate secrets management

### 2. Deployment Best Practices

#### Configuration Deployment
- **Environment Consistency**: Maintain configuration consistency
- **Rollback Capability**: Enable configuration rollback
- **Gradual Rollout**: Gradually deploy configuration changes
- **Monitoring**: Monitor configuration changes

#### Configuration Updates
- **Hot Reloading**: Support configuration hot reloading
- **Graceful Updates**: Update configuration gracefully
- **Validation**: Validate updates before applying
- **Rollback**: Enable rollback of configuration changes

## Configuration Migration

### 1. Version Management

#### Configuration Versioning
- **Semantic Versioning**: Use semantic versioning for configurations
- **Backward Compatibility**: Maintain backward compatibility
- **Migration Scripts**: Provide configuration migration scripts
- **Change Logs**: Maintain configuration change logs

#### Schema Evolution
- **Schema Validation**: Validate configuration schema
- **Migration Paths**: Define configuration migration paths
- **Compatibility Testing**: Test configuration compatibility
- **Deprecation**: Handle deprecated configuration options

### 2. Migration Strategies

#### Zero-Downtime Migration
- **Gradual Migration**: Migrate configuration gradually
- **Feature Flags**: Use feature flags for configuration changes
- **A/B Testing**: Test new configurations with A/B testing
- **Rollback Plans**: Maintain configuration rollback plans

## Future Configuration Enhancements

### 1. Advanced Configuration Features
- **Dynamic Configuration**: Real-time configuration updates
- **Machine Learning**: ML-based configuration optimization
- **Auto-Tuning**: Automatic configuration tuning
- **Predictive Configuration**: Predictive configuration management

### 2. Enhanced Management Tools
- **Configuration UI**: User interface for configuration management
- **Dashboard**: Configuration monitoring dashboard
- **API**: Configuration management API
- **CLI**: Command-line configuration tools

This configuration management system ensures that the RAG system can operate effectively across different environments while maintaining security, performance, and reliability.