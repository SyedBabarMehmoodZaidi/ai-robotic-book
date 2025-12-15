# RAG System Developer Documentation

This document provides comprehensive technical documentation for developers working with the Retrieval-Augmented Generation (RAG) system integrated into the Physical AI & Humanoid Robotics book.

## Overview

The RAG (Retrieval-Augmented Generation) system integrates AI-powered question answering capabilities directly into the Docusaurus-based book interface. This system allows users to ask questions about book content and receive AI-generated responses based on the book's information.

## Architecture

### System Components

#### Frontend Components
```
src/components/
├── QueryInterface/          # Main query interface
│   ├── QueryInterface.js    # React component
│   └── QueryInterface.module.css # Styling
├── SelectedTextCapture/     # Text selection capture
│   ├── SelectedTextCapture.js # React component
│   └── SelectedTextCapture.module.css # Styling (if needed)
└── RAGValidation/           # Validation tools
    ├── RAGValidation.js     # Validation component
    └── RAGValidation.module.css # Styling
```

#### Static Assets
```
static/js/
├── api-client.js           # API communication layer
├── config.js               # Configuration parameters
└── backend-verification.js # Backend connectivity verification
```

### Data Flow

```
User Interaction → SelectedTextCapture → QueryInterface → RAGApiClient → Backend API
     ↓
Response Display ← QueryInterface ← RAGApiClient ← Backend API
```

## Component Details

### 1. QueryInterface Component

#### Purpose
- Provides the main interface for users to ask questions
- Handles form submission and response display
- Integrates with selected text functionality

#### Props
- None (self-contained component)

#### State Management
- `query`: Current query text
- `response`: API response object
- `isLoading`: Loading state indicator
- `error`: Error messages
- `selectedText`: Text selected on the page

#### API Integration
- Uses global `window.RAGApiClient` for communication
- Calls `RAGApiClient.query(query, selectedText)` method
- Handles success and error responses

### 2. SelectedTextCapture Component

#### Purpose
- Captures text selected by users on the page
- Validates selected text length
- Communicates selected text to parent components

#### Props
- `onTextSelected`: Callback function when text is selected

#### Event Handling
- Listens for `mouseup` and `keyup` events
- Uses `window.getSelection()` to capture selected text
- Validates text length against configuration parameters

### 3. RAGApiClient

#### Purpose
- Handles all API communication with the backend
- Implements retry mechanisms and error handling
- Manages timeouts and request validation

#### Methods
- `query(queryText, selectedText)`: Submit query to backend
- `healthCheck()`: Check backend health status
- `testConnection()`: Test connectivity to backend

#### Configuration
- Uses `window.RAGConfig` for parameters
- Implements timeout and retry logic
- Validates input parameters

## API Contracts

### Backend Endpoints

#### Query Endpoint: `POST /query`
```json
{
  "query_text": "string, required - The question or query to ask the AI",
  "selected_text": "string or null, optional - Text selected by the user on the page"
}
```

Response:
```json
{
  "response": {
    "response_text": "string - The AI-generated response to the query",
    "source_context": "array of strings - Source documents/chunks used in generating the response",
    "confidence_score": "float - Confidence score between 0.0 and 1.0"
  }
}
```

#### Health Check Endpoint: `GET /health`
Response:
```json
{
  "status": "string - Health status (e.g., 'healthy', 'degraded', 'unhealthy')",
  "version": "string - API version",
  "timestamp": "string - ISO 8601 timestamp of the health check",
  "details": "object - Additional health check details (optional)"
}
```

### Frontend Configuration

#### Configuration Parameters
```javascript
const RAGConfig = {
  BACKEND_API_URL: "http://localhost:8000",  // Default backend URL
  DEVELOPMENT_MODE: true,                     // Development mode flag
  REQUEST_TIMEOUT: 30000,                     // Request timeout in milliseconds
  MAX_RETRIES: 2,                            // Maximum retry attempts
  RETRY_DELAY: 1000,                         // Delay between retries (ms)
  MIN_QUERY_LENGTH: 3,                       // Minimum query length
  MAX_QUERY_LENGTH: 2000,                    // Maximum query length
  MIN_SELECTED_TEXT_LENGTH: 10,              // Minimum selected text length
  MAX_SELECTED_TEXT_LENGTH: 5000             // Maximum selected text length
};
```

## Implementation Details

### Docusaurus Integration

#### Component Loading
The RAG components are loaded via MDX imports in book pages:
```md
import QueryInterface from '@site/src/components/QueryInterface/QueryInterface';

<QueryInterface />
```

#### Plugin Integration
Custom Docusaurus plugin injects required scripts into all pages:
```javascript
// In docusaurus.config.js
plugins: [
  async function myPlugin(context, options) {
    return {
      name: 'custom-backend-verification',
      injectHtmlTags() {
        return {
          headTags: [
            { tagName: 'script', attributes: { src: '/js/config.js' } },
            { tagName: 'script', attributes: { src: '/js/api-client.js' } },
            { tagName: 'script', attributes: { src: '/js/backend-verification.js' } },
          ],
        };
      },
    };
  },
],
```

### Styling System

#### CSS Modules
Each component uses CSS modules for isolated styling:
- Component-specific styles
- No global CSS conflicts
- Easy maintenance and updates

### Error Handling

#### Frontend Error Handling
- Input validation on both client and API client
- Network error handling with retry mechanisms
- User-friendly error messages
- Graceful degradation when backend is unavailable

#### Backend Error Handling
- Structured error responses
- Proper HTTP status codes
- Comprehensive error logging
- Fallback mechanisms

## Development Workflow

### Local Development Setup

#### Prerequisites
- Node.js 16+ installed
- npm or yarn package manager
- Docusaurus development environment

#### Setup Steps
```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
npm install

# Start development server
npm start

# The RAG components will be automatically available
```

#### Backend Requirements
- RAG backend service running on localhost:8000
- Proper CORS configuration for local development
- Health check endpoint accessible

### Testing Strategy

#### Unit Tests
Located in `tests/rag-components.test.js`:
- Component rendering tests
- User interaction simulations
- State management validation
- API integration mocks

#### Integration Tests
- Component-to-component communication
- API client integration
- Configuration validation
- End-to-end workflows

#### Testing Commands
```bash
# Run all tests
npm test

# Run tests with coverage
npm run test:coverage

# Run tests in watch mode
npm run test:watch
```

### Building for Production

#### Build Process
```bash
# Set environment variables
export NODE_ENV=production
export BACKEND_API_URL=https://your-backend-url.com

# Build the site
npm run build
```

#### Production Considerations
- HTTPS configuration for secure communication
- Proper CORS headers for production domain
- Performance optimization for production assets
- Error tracking and monitoring setup

## Performance Optimization

### Client-Side Optimizations

#### Code Splitting
- React components are isolated
- CSS modules prevent style conflicts
- Minimal bundle sizes through tree shaking

#### API Optimizations
- Timeout handling with Promise.race
- Retry mechanisms with exponential backoff
- Input validation to prevent unnecessary requests
- Caching strategies (future enhancement)

### Backend Performance Considerations

#### Response Times
- Target: < 3 seconds for typical queries
- Timeout: 30 seconds (configurable)
- Retry mechanism for failed requests
- Connection pooling for multiple requests

## Security Considerations

### Frontend Security

#### Input Validation
- Client-side validation of query parameters
- Selected text length validation
- XSS prevention through React rendering
- Secure configuration management

#### Communication Security
- HTTPS recommended for production
- Secure API endpoint configuration
- Proper error message sanitization
- No sensitive data in client configuration

### Backend Security

#### API Security
- Input sanitization on backend
- Rate limiting implementation
- Authentication mechanisms (if required)
- Proper error response handling

## Deployment Configuration

### Environment Variables

#### Required Variables
- `BACKEND_API_URL`: RAG backend API endpoint
- `NODE_ENV`: Environment indicator (development/production)

#### Optional Variables
- `REQUEST_TIMEOUT`: API request timeout (default: 30000ms)
- `MAX_RETRIES`: Maximum retry attempts (default: 2)
- `RETRY_DELAY`: Delay between retries (default: 1000ms)

### Production Deployment

#### Static Asset Deployment
- Build output in `build/` directory
- CDN-friendly asset organization
- Compression enabled (gzip/brotli)
- Cache headers configuration

#### Backend Integration
- Proper CORS configuration
- Health check endpoint accessibility
- Error logging and monitoring
- Security headers configuration

## Troubleshooting

### Common Issues

#### Backend Connectivity
**Issue**: "Failed to connect to AI service"
**Solutions**:
- Verify backend service is running on configured URL
- Check network connectivity
- Verify CORS configuration
- Check firewall settings

#### Component Not Loading
**Issue**: QueryInterface component not appearing
**Solutions**:
- Verify MDX import syntax
- Check component file paths
- Verify Docusaurus build process
- Check browser console for errors

#### Slow Response Times
**Issue**: API responses taking too long
**Solutions**:
- Check backend performance
- Verify network connectivity
- Adjust timeout configuration
- Monitor backend resource usage

### Debugging Tools

#### Browser Developer Tools
- Network tab for API communication
- Console for error messages
- Elements tab for component structure
- Performance tab for timing analysis

#### Logging Configuration
- Development: `LOG_LEVEL=debug`
- Production: `LOG_LEVEL=error`
- Error tracking services integration
- Performance monitoring tools

## Extending the System

### Adding New Features

#### Component Extensions
- Create new components in `src/components/`
- Follow existing component patterns
- Use CSS modules for styling
- Implement proper error handling

#### API Extensions
- Add new endpoints to backend
- Update API client methods
- Modify configuration parameters
- Update documentation accordingly

### Integration Points

#### Custom Components
- Import RAG components in MDX files
- Pass custom props if needed
- Integrate with existing page layouts
- Maintain consistent styling

#### Configuration Extensions
- Add new configuration parameters
- Update validation logic
- Document new parameters
- Provide default values

## Maintenance and Updates

### Code Quality

#### Linting
- ESLint configuration included
- React-specific linting rules
- Consistent code formatting
- Automated linting in CI/CD

#### Testing Requirements
- Minimum 80% code coverage
- All tests must pass before deployment
- Integration tests for new features
- Performance tests for critical paths

### Monitoring and Analytics

#### Performance Metrics
- API response times
- Page load performance
- User interaction analytics
- Error rate monitoring

#### Health Checks
- Backend service availability
- Component functionality verification
- Configuration validation
- Automated alerting for issues

This developer documentation provides comprehensive guidance for maintaining, extending, and troubleshooting the RAG system integration with the Physical AI & Humanoid Robotics book.