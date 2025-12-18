from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
# Load model and tokenizer
model_id = "LiquidAI/LFM2-1.2B-Tool"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype="bfloat16",
#    attn_implementation="flash_attention_2" <- uncomment on compatible GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Creating prompt...")
# Create message
system_prompt = """List of tools:

<|tool_list_start|>[
[
{
"name": "check_criminal_history",
"description": "Access law enforcement databases to retrieve criminal records for a suspect or person of interest.",
"parameters": {
"type": "object",
"properties": {
"suspect_name": {
"type": "string",
"description": "Full name of the individual being investigated."
},
"jurisdiction": {
"type": "string",
"description": "Geographic region or agency responsible for the criminal records (e.g., 'New York PD', 'FBI')."
},
"date_of_birth": {
"type": "string",
"format": "date",
"description": "Date of birth in YYYY-MM-DD format for verification."
}
},
"required": [
"suspect_name",
"date_of_birth"
]
}
},
{
"name": "analyze_crime_scene_image",
"description": "Perform forensic analysis on digital images from a crime scene to detect evidence or patterns.",
"parameters": {
"type": "object",
"properties": {
"image_path": {
"type": "string",
"description": "File path or URL where the crime scene image is stored."
},
"analysis_type": {
"type": "string",
"enum": [
"weapon_detection",
"blood_spatter_pattern",
"footprint_analysis"
],
"description": "Specific type of analysis to perform on the image."
},
"confidence_threshold": {
"type": "float",
"minimum": 0.0,
"maximum": 1.0,
"description": "Minimum confidence level (0-1) for reported findings."
}
},
"required": [
"image_path",
"analysis_type"
]
}
},
{
"name": "retrieve_surveillance_video",
"description": "Access and retrieve video footage from security cameras in a specific location during a defined timeframe.",
"parameters": {
"type": "object",
"properties": {
"location": {
"type": "string",
"description": "Address or landmark where the surveillance camera is located."
},
"start_datetime": {
"type": "string",
"format": "date-time",
"description": "Start time for the video search (ISO 8601 format)."
},
"end_datetime": {
"type": "string",
"format": "date-time",
"description": "End time for the video search (ISO 8601 format)."
},
"camera_id": {
"type": "string",
"description": "Specific camera identifier if multiple cameras exist at the location."
}
},
"required": [
"location",
"start_datetime",
"end_datetime"
]
}
},
{
"name": "get_dna_match",
"description": "Query a forensic DNA database to identify potential matches for a biological sample collected at a crime scene.",
"parameters": {
"type": "object",
"properties": {
"sample_id": {
"type": "string",
"description": "Unique identifier for the biological sample in the evidence database."
},
"database_name": {
"type": "string",
"enum": [
"CODIS",
"state_local",
"international"
],
"description": "The specific DNA database to query (e.g., CODIS for federal databases)."
}
},
"required": [
"sample_id",
"database_name"
]
}
},
{
"name": "retrieve_vehicle_vin_info",
"description": "Access a vehicle registration database to obtain details about a vehicle's ownership history, accident records, and mechanical specifications using its VIN.",
"parameters": {
"type": "object",
"properties": {
"vin": {
"type": "string",
"description": "17-character Vehicle Identification Number of the vehicle."
},
"jurisdiction": {
"type": "string",
"description": "Geographic region or agency responsible for the vehicle's registration (e.g., 'State DMV', 'National Vehicle Registry')."
}
},
"required": [
"vin"
]
}
},
{
"name": "verify_alibi_with_public_records",
"description": "Cross-reference a suspect's claimed location with public event or attendance records, including restaurant reservations, library checkouts, and public transportation logs.",
"parameters": {
"type": "object",
"properties": {
"person_name": {
"type": "string",
"description": "Full name of the individual being verified."
},
"date": {
"type": "string",
"format": "date",
"description": "Date of the claimed alibi (YYYY-MM-DD format)."
},
"location": {
"type": "string",
"description": "Specific location where the alibi was claimed to have occurred."
},
"confidence_threshold": {
"type": "float",
"minimum": 0.0,
"maximum": 1.0,
"description": "Minimum confidence level (0-1) for matching records."
}
},
"required": [
"person_name",
"date",
"location"
]
}
},
{
"name": "analyze_toxicology_report",
"description": "Query a medical forensic database to identify substances present in a biological sample, including drugs, alcohol, and toxins. Returns concentration levels and potential lethal thresholds.",
"parameters": {
"type": "object",
"properties": {
"sample_id": {
"type": "string",
"description": "Unique identifier for the toxicology sample in the evidence database."
},
"test_type": {
"type": "string",
"enum": [
"alcohol",
"drugs",
"poisons",
"medications"
],
"description": "Type of toxicological analysis to perform."
},
"confidence_threshold": {
"type": "float",
"minimum": 0.0,
"maximum": 1.0,
"description": "Minimum confidence level (0-1) for reported findings."
}
},
"required": [
"sample_id",
"test_type"
]
}
},
{
"name": "check_suspect_travel_history",
"description": "Retrieve transportation records to determine a suspect's movements during a specific timeframe. This tool queries national and local transit databases, including flight, rail, and toll road records.",
"parameters": {
"type": "object",
"properties": {
"suspect_id": {
"type": "string",
"description": "Unique identifier for the suspect in law enforcement systems."
},
"start_datetime": {
"type": "string",
"format": "date-time",
"description": "Start time for the travel history search (ISO 8601 format)."
},
"end_datetime": {
"type": "string",
"format": "date-time",
"description": "End time for the travel history search (ISO 8601 format)."
}
},
"required": [
"suspect_id",
"start_datetime",
"end_datetime"
]
}
},
{
"name": "check_outstanding_warrants",
"description": "Query law enforcement databases to check for active warrants issued against a suspect, including arrest, bail, or restraining orders.",
"parameters": {
"type": "object",
"properties": {
"suspect_id": {
"type": "string",
"description": "Unique identifier for the suspect in law enforcement systems (e.g., national ID or case number)."
},
"jurisdiction": {
"type": "string",
"description": "Geographic region or agency responsible for issuing the warrant (e.g., 'Local Police Department', 'FBI')."
}
},
"required": [
"suspect_id"
]
}
},
{
"name": "analyze_social_media_activity",
"description": "Scrape and analyze social media content (e.g., posts, messages, geotags) of a suspect or victim to identify patterns, connections, or evidence of threats.",
"parameters": {
"type": "object",
"properties": {
"platform": {
"type": "string",
"enum": [
"Facebook",
"Twitter",
"Instagram",
"WhatsApp",
"TikTok"
],
"description": "Social media platform to analyze (e.g., 'Facebook')."
},
"user_handle": {
"type": "string",
"description": "Username or profile handle of the suspect or victim."
},
"start_date": {
"type": "string",
"format": "date",
"description": "Start date for content analysis (YYYY-MM-DD)."
},
"end_date": {
"type": "string",
"format": "date",
"description": "End date for content analysis (YYYY-MM-DD)."
}
},
"required": [
"platform",
"user_handle",
"start_date",
"end_date"
]
}
},
{
"name": "check_weapon_registration",
"description": "Access firearms registries to confirm if a suspect or victim owns a specific weapon, including its history of transfers or incidents.",
"parameters": {
"type": "object",
"properties": {
"firearm_serial": {
"type": "string",
"description": "10-15 digit serial number engraved on the firearm (e.g., '1234567890ABCDEF')."
},
"jurisdiction": {
"type": "string",
"description": "Geographic region or agency responsible for firearm registration (e.g., 'State Firearms Bureau')."
}
},
"required": [
"firearm_serial"
]
}
},
{
"name": "extract_digital_evidence",
"description": "Retrieve and decrypt data from a suspect's digital devices (e.g., smartphones, laptops) for forensic analysis, including messages, call logs, and location history.",
"parameters": {
"type": "object",
"properties": {
"device_serial": {
"type": "string",
"description": "Serial number or unique identifier of the digital device (e.g., iPhone UDID, Android IMEI)."
},
"start_datetime": {
"type": "string",
"format": "date-time",
"description": "Start time for data extraction (ISO 8601 format)."
},
"end_datetime": {
"type": "string",
"format": "date-time",
"description": "End time for data extraction (ISO 8601 format)."
}
},
"required": [
"device_serial",
"start_datetime",
"end_datetime"
]
}
},
{
"name": "verify_suspect_employment_records",
"description": "Access employment databases to confirm the suspect's work history, job responsibilities, and potential motives such as workplace conflicts or financial distress.",
"parameters": {
"type": "object",
"properties": {
"suspect_id": {
"type": "string",
"description": "Unique identifier for the suspect in law enforcement systems."
},
"start_date": {
"type": "string",
"format": "date",
"description": "Start date for employment verification (YYYY-MM-DD)."
},
"end_date": {
"type": "string",
"format": "date",
"description": "End date for employment verification (YYYY-MM-DD)."
}
},
"required": [
"suspect_id",
"start_date",
"end_date"
]
}
}
]<|tool_list_end|>

If you call a function, output also a message for the user that the function is being called. The message should be detailed."""
message = [
  {"role": "system", "content": system_prompt},
  {"role": "user", "content": "Check the criminal history for John Doe, born on 1985-03-15, under the jurisdiction of Los Angeles PD."}
]

# Generate answer
input_ids = tokenizer.apply_chat_template(
    message,
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
).to(model.device)

output = model.generate(
    input_ids,
    do_sample=False,
    max_new_tokens=512,
)

print(tokenizer.decode(output[0], skip_special_tokens=False))

# outputs = model.generate(**inputs.to(model.device), max_new_tokens=256)
# final = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)