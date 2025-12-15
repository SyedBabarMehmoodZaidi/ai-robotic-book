# RAG Validation Tests

This document outlines the validation tests for the Retrieval-Augmented Generation (RAG) system integration with the Physical AI & Humanoid Robotics book.

## Test Overview

The RAG validation tests ensure that the frontend integration with the backend RAG service functions correctly and provides reliable responses to user queries.

## Test Categories

### 1. Functional Tests

#### 1.1 Query Interface Tests

**Test ID:** FT-RAG-001
**Test Name:** Query Submission
**Objective:** Verify that users can submit queries through the interface
**Preconditions:**
- QueryInterface component is loaded
- Backend service is accessible
- User has entered text in the query field

**Test Steps:**
1. Enter a query in the text area (e.g., "What is ROS 2?")
2. Click the "Ask AI" button
3. Verify loading indicator appears
4. Verify response is displayed after processing

**Expected Results:**
- Query is sent to backend API
- Loading indicator is shown during processing
- Response is displayed in the response container
- Source context and confidence score are shown if available

---

**Test ID:** FT-RAG-002
**Test Name:** Selected Text Integration
**Objective:** Verify that selected text is captured and sent with queries
**Preconditions:**
- SelectedTextCapture component is active
- User has selected text on the page
- QueryInterface component is available

**Test Steps:**
1. Select text on the book page (minimum 10 characters)
2. Verify selected text preview appears in QueryInterface
3. Enter a query related to the selected text
4. Submit the query
5. Verify selected text is included in the API request

**Expected Results:**
- Selected text is captured and displayed as preview
- Selected text is sent to backend with the query
- AI response considers the selected text context

---

**Test ID:** FT-RAG-003
**Test Name:** Response Display
**Objective:** Verify that responses are properly displayed
**Preconditions:**
- Query has been submitted successfully
- Response has been received from backend

**Test Steps:**
1. Submit a query that returns a response with source context
2. Verify response text is displayed
3. Verify source context is shown
4. Verify confidence score is displayed

**Expected Results:**
- Response text is displayed in readable format
- Source context shows relevant document references
- Confidence score is displayed as percentage
- All elements are properly styled

### 2. Integration Tests

#### 2.1 API Communication Tests

**Test ID:** IT-RAG-001
**Test Name:** API Client Functionality
**Objective:** Verify that the API client communicates correctly with the backend
**Preconditions:**
- API client is initialized
- Backend service is running and accessible

**Test Steps:**
1. Call `RAGApiClient.query()` with valid parameters
2. Verify request is sent to correct endpoint
3. Verify response is properly parsed
4. Call `RAGApiClient.healthCheck()` to verify connectivity

**Expected Results:**
- Query requests are sent to `/query` endpoint
- Health check requests are sent to `/health` endpoint
- Responses are properly parsed as JSON
- Error handling works for invalid responses

---

**Test ID:** IT-RAG-002
**Test Name:** Configuration Integration
**Objective:** Verify that configuration parameters are properly used
**Preconditions:**
- Configuration file is loaded
- API client has access to configuration

**Test Steps:**
1. Verify API client uses configured base URL
2. Test timeout functionality with slow response
3. Verify retry mechanism works for failed requests
4. Test input validation with various query lengths

**Expected Results:**
- API client uses correct base URL from config
- Requests timeout after configured duration
- Failed requests are retried according to config
- Input validation follows configured rules

#### 2.2 Component Integration Tests

**Test ID:** IT-RAG-003
**Test Name:** Component Communication
**Objective:** Verify that frontend components communicate correctly
**Preconditions:**
- QueryInterface and SelectedTextCapture components are loaded
- Components are properly connected

**Test Steps:**
1. Select text on the page
2. Verify SelectedTextCapture updates QueryInterface state
3. Submit a query with selected text
4. Verify both components work together

**Expected Results:**
- Selected text is passed from SelectedTextCapture to QueryInterface
- Both components maintain proper state
- Query includes selected text when available

### 3. Error Handling Tests

#### 3.1 Client-Side Validation Tests

**Test ID:** ET-RAG-001
**Test Name:** Query Validation
**Objective:** Verify that client-side validation works
**Preconditions:**
- QueryInterface component is loaded

**Test Steps:**
1. Enter a query with less than 3 characters
2. Attempt to submit the query
3. Enter a query with more than 2000 characters
4. Attempt to submit the query

**Expected Results:**
- Short queries are rejected before API call
- Long queries are rejected before API call
- Appropriate error messages are shown

---

**Test ID:** ET-RAG-002
**Test Name:** Selected Text Validation
**Objective:** Verify that selected text validation works
**Preconditions:**
- SelectedTextCapture component is active

**Test Steps:**
1. Select text with less than 10 characters
2. Verify it's not captured as selected text
3. Select text with more than 5000 characters
4. Verify it's not captured as selected text

**Expected Results:**
- Short selections are ignored
- Long selections are ignored
- Only valid-length selections are captured

#### 3.2 Network Error Tests

**Test ID:** ET-RAG-003
**Test Name:** Backend Connectivity
**Objective:** Verify error handling when backend is unavailable
**Preconditions:**
- Backend service is stopped or unreachable

**Test Steps:**
1. Stop the backend service
2. Submit a query
3. Verify error message is displayed
4. Restart backend and verify functionality returns

**Expected Results:**
- Appropriate error message is shown when backend is unreachable
- Retry mechanism attempts to recover
- Functionality returns when backend is available

### 4. Performance Tests

#### 4.1 Response Time Tests

**Test ID:** PT-RAG-001
**Test Name:** Query Response Time
**Objective:** Verify acceptable response times
**Preconditions:**
- Backend service is running normally
- Valid query is prepared

**Test Steps:**
1. Submit a query
2. Measure time from submission to response
3. Repeat with different query types
4. Verify response times are within acceptable limits

**Expected Results:**
- Response times are less than 30 seconds (timeout limit)
- Average response time is under 10 seconds
- No timeouts occur under normal conditions

### 5. User Experience Tests

#### 5.1 Interface Usability Tests

**Test ID:** UX-RAG-001
**Test Name:** Interface Accessibility
**Objective:** Verify the interface is user-friendly
**Preconditions:**
- QueryInterface component is loaded

**Test Steps:**
1. Verify all interface elements are visible
2. Test keyboard navigation
3. Verify button states (enabled/disabled)
4. Test responsive design on different screen sizes

**Expected Results:**
- All elements are properly displayed
- Interface is accessible via keyboard
- Button states change appropriately
- Interface adapts to different screen sizes

## Test Execution

### Automated Testing
- Unit tests for individual components
- Integration tests for API client
- End-to-end tests for complete workflows

### Manual Testing
- User acceptance testing
- Cross-browser compatibility testing
- Accessibility testing

## Success Criteria

A test is considered successful if:
- All expected results are achieved
- No critical errors occur during execution
- Performance metrics are met
- User experience requirements are satisfied

## Test Environment

- **Frontend:** Docusaurus documentation site
- **Backend:** RAG service running on localhost:8000
- **Browsers:** Chrome, Firefox, Safari, Edge
- **Devices:** Desktop, tablet, mobile

## Test Data

### Valid Test Queries
- "What is ROS 2?"
- "Explain the differences between Gazebo and Unity simulation"
- "How does VSLAM work in robotics?"
- "What are the key components of NVIDIA Isaac?"

### Valid Selected Text Examples
- Excerpts from module documentation (10-5000 characters)
- Code snippets with explanations
- Technical definitions and concepts

## Reporting

Test results should be documented with:
- Test ID and name
- Execution date and time
- Environment details
- Pass/fail status
- Any issues or anomalies observed
- Screenshots if necessary