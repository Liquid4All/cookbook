def get_system_prompt(version: str = "v1") -> str:
    """Returns the system prompt based on the specified version."""
    if version == "v1":
        return system_prompt_v1
    elif version == "v2":
        return system_prompt_v2
    else:
        raise ValueError(f"Unsupported version: {version}")
    

system_prompt_v1 = """Return data as a JSON object with the following schema:

The output should be an array of objects. Each object represents an extracted medical entity and must contain:

1. "text" (string, required): The extracted medical term, condition name, medication name, or measurement name
2. "category" (string, required): One of "MEDICAL_CONDITION", "MEDICATION", or "MEASUREMENT"
3. Additional fields based on category:
    - If category is "MEDICATION": include "dosage" object with "text" field containing dosage information
    - If category is "MEASUREMENT": include "value" object with "text" field containing the measurement value and units
    - If category is "MEDICAL_CONDITION": no additional fields required

JSON schema:
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "required": ["text", "category"],
    "properties": {
      "text": {
        "type": "string",
        "description": "The extracted medical term, condition name, medication name, or measurement name"
      },
      "category": {
        "type": "string",
        "enum": ["MEDICAL_CONDITION", "MEDICATION", "MEASUREMENT"],
        "description": "The type of medical entity"
      },
      "dosage": {
        "type": "object",
        "properties": {
          "text": {
            "type": "string",
            "description": "Dosage information including amount and frequency"
          }
        },
        "required": ["text"]
      },
      "value": {
        "type": "object",
        "properties": {
          "text": {
            "type": "string",
            "description": "The measurement value with units"
          }
        },
        "required": ["text"]
      }
    },
    "allOf": [
      {
        "if": {
          "properties": { "category": { "const": "MEDICATION" } }
        },
        "then": {
          "required": ["dosage"]
        }
      },
      {
        "if": {
          "properties": { "category": { "const": "MEASUREMENT" } }
        },
        "then": {
          "required": ["value"]
        }
      }
    ]
  }
}
"""

system_prompt_v2 = """Identify and extract medical entities from the text. Present as a JSON array of objects.

Medical Entities:
Extract all relevant medical entities and categorize them appropriately.

Entity Structure:
text: The actual text/phrase of the medical entity as it appears in the document.
category: The type of medical entity. Must be one of:
  - MEDICAL_CONDITION: Diseases, symptoms, diagnoses, or health conditions.
  - MEDICATION: Drugs, prescriptions, or therapeutic substances.
  - MEASUREMENT: Vital signs, lab values, or quantifiable health metrics.

Category-Specific Fields:

For MEDICATION entities only:
dosage: An object containing dosage information.
  text: The dosage amount and frequency (e.g., "10mg twice daily", "500mg as needed").

For MEASUREMENT entities only:
value: An object containing the measured value.
  text: The numerical value with units (e.g., "120/80 mmHg", "98.6°F", "150 mg/dL").

Notes:
- Each entity should be a separate object in the array.
- Include dosage field only for MEDICATION category.
- Include value field only for MEASUREMENT category.
- Do not include dosage or value fields for MEDICAL_CONDITION category.
"""