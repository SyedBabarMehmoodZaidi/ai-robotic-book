# Research: RAG Backend Integration for Frontend Book Interface

**Feature**: 1-rag-frontend-integration
**Created**: 2025-12-15
**Status**: Complete
**Author**: Claude

## Research Tasks

### Task 1: API Contract Research

**Decision**: The RAG backend from Spec-3 exposes a POST /query endpoint that accepts query requests and returns AI-generated responses.

**Rationale**: This endpoint was identified in the existing backend implementation and matches the functional requirements for query submission.

**Alternatives considered**:
- Creating a new endpoint: Would require backend changes, violating the constraint of not modifying Spec-3 backend
- Using a different communication pattern: HTTP/JSON was specified in the requirements

**Findings**:
- Endpoint: POST /query
- Request format: JSON object with query_text and optional selected_text fields
- Response format: JSON object with response_text, source_context, confidence_score, and other metadata
- Example request: `{"query_text": "What is AI?", "selected_text": "Optional selected text"}`

### Task 2: Docusaurus Customization Patterns

**Decision**: Use Docusaurus MDX components to add the query interface to book pages, with custom CSS for styling that matches the existing theme.

**Rationale**: MDX allows embedding React components in Markdown content, which is the standard approach for extending Docusaurus functionality without breaking existing features.

**Alternatives considered**:
- Modifying Docusaurus theme files: Would be harder to maintain and could break on updates
- Adding script tags directly to pages: Less maintainable and doesn't follow Docusaurus patterns
- Creating a custom plugin: Overkill for this simple integration

**Findings**:
- Docusaurus supports MDX components that can be embedded in Markdown files
- Components can be created in src/components/ directory
- CSS can be customized using Docusaurus' CSS variables and theme system
- Custom components can access page context and add functionality

### Task 3: Text Selection API Implementation

**Decision**: Use the browser's Selection API combined with document.getSelection() and window.getSelection() methods to capture selected text.

**Rationale**: The Selection API is well-supported across modern browsers and provides reliable text selection capture functionality.

**Alternatives considered**:
- Manual text selection tracking: More complex and error-prone
- Third-party libraries: Would add unnecessary dependencies
- Mouse event tracking: Less reliable than built-in Selection API

**Findings**:
- Use `window.getSelection().toString()` to get currently selected text
- Add event listeners for 'mouseup' and 'keyup' events to detect text selection
- Can get more detailed selection information using Selection API methods
- Need to handle edge cases like selections across multiple elements

### Task 4: Frontend Communication Patterns

**Decision**: Use the Fetch API for HTTP/JSON communication with proper error handling and timeout management.

**Rationale**: Fetch API is the modern standard for making HTTP requests in browsers, supports JSON natively, and provides good error handling capabilities.

**Alternatives considered**:
- XMLHttpRequest: Older technology, Fetch API is preferred
- Axios library: Would add an unnecessary dependency for simple requests
- jQuery AJAX: Would add unnecessary dependency and jQuery is not used elsewhere

**Findings**:
- Use async/await pattern for cleaner code
- Implement proper error handling for network failures
- Add timeout handling to prevent hanging requests
- Include appropriate headers (Content-Type: application/json)
- Handle CORS appropriately for local development

## API Integration Details

### Backend API Contract (from existing Spec-3 implementation)

**POST /query**
- Request body: `{"query_text": "string", "selected_text": "string"}`
- Response body: `{"response": {"response_text": "string", "source_context": ["string"], "confidence_score": number}, "request_id": "string", "status_code": number, "timestamp": "string", "processing_time": number}`
- Headers: Content-Type: application/json
- Expected response time: Under 5 seconds for 90% of requests

### Frontend Implementation Patterns

**Query Component Structure**:
- Input field for user queries
- Submit button that triggers API call
- Display area for AI responses
- Selected text indicator
- Loading state during API requests
- Error display for failed requests

**Communication Flow**:
1. User enters query in input field
2. If text is selected on page, capture it automatically
3. Send POST request to backend with query and selected text
4. Show loading state to user
5. Receive response and display in response area
6. Handle errors gracefully with user-friendly messages

## Security Considerations

**Input Validation**:
- Sanitize user queries before sending to backend
- Validate response content before displaying
- Implement proper error handling to prevent XSS

**CORS Configuration**:
- Backend should allow requests from localhost during development
- Production configuration will depend on deployment setup

## Performance Considerations

**Optimization Strategies**:
- Implement loading indicators to improve perceived performance
- Cache responses for identical queries (optional enhancement)
- Use efficient DOM updates for response display
- Implement request timeout to prevent hanging calls

## Browser Compatibility

**Target Browsers**: Chrome, Firefox, Safari, Edge (modern versions)

**API Support**:
- Fetch API: Supported in all target browsers
- Selection API: Supported in all target browsers
- Async/await: Supported in all target browsers
- Modern CSS: Supported in all target browsers