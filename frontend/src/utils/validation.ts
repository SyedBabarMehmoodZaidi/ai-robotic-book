// Input validation utility for the application

export interface ValidationRule {
  rule: (value: any) => boolean;
  message: string;
}

export interface ValidationResult {
  isValid: boolean;
  errors: string[];
}

export class Validation {
  /**
   * Validate query text length and content
   */
  static validateQueryText(queryText: string): ValidationResult {
    const errors: string[] = [];

    if (!queryText || queryText.trim().length === 0) {
      errors.push('Query text is required');
    } else if (queryText.length > 2000) {
      errors.push('Query text must be 2000 characters or less');
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  /**
   * Validate context text length
   */
  static validateContextText(contextText?: string): ValidationResult {
    if (!contextText) {
      return { isValid: true, errors: [] };
    }

    const errors: string[] = [];

    if (contextText.length > 10000) {
      errors.push('Context text must be 10000 characters or less');
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  /**
   * Validate query type
   */
  static validateQueryType(queryType?: string): ValidationResult {
    const validTypes = ['general', 'context-specific'];
    const errors: string[] = [];

    if (!queryType) {
      errors.push('Query type is required');
    } else if (!validTypes.includes(queryType)) {
      errors.push(`Query type must be one of: ${validTypes.join(', ')}`);
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  /**
   * Validate selected text
   */
  static validateSelectedText(selectedText?: string): ValidationResult {
    if (!selectedText) {
      return { isValid: true, errors: [] };
    }

    const errors: string[] = [];

    if (selectedText.length > 10000) {
      errors.push('Selected text is too long (max 10000 characters)');
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  /**
   * Validate the entire query object
   */
  static validateQuery(query: any): ValidationResult {
    const errors: string[] = [];

    // Validate query text
    const queryTextValidation = this.validateQueryText(query.query_text);
    if (!queryTextValidation.isValid) {
      errors.push(...queryTextValidation.errors);
    }

    // Validate context text if provided
    const contextValidation = this.validateContextText(query.context_text);
    if (!contextValidation.isValid) {
      errors.push(...contextValidation.errors);
    }

    // Validate query type if provided
    if (query.query_type) {
      const typeValidation = this.validateQueryType(query.query_type);
      if (!typeValidation.isValid) {
        errors.push(...typeValidation.errors);
      }
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  /**
   * Generic validation function with custom rules
   */
  static validate(value: any, rules: ValidationRule[]): ValidationResult {
    const errors: string[] = [];

    for (const rule of rules) {
      if (!rule.rule(value)) {
        errors.push(rule.message);
      }
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }
}

export default Validation;