import os
import json
import datetime
import requests
import re
import google.generativeai as genai

# ==========================================
# ⚙️ CONFIGURATION & KEYWORDS
# ==========================================
SEARCH_KEYWORDS = "DevOps SRE Reliability Platform MLOps Infrastructure"
TODAY = datetime.datetime.now().strftime("%Y-%m-%d")
API_KEY = os.environ.get("GEMINI_API_KEY")

for folder in["jobs", "aggregated", "reports"]:
    os.makedirs(folder, exist_ok=True)

# ==========================================
# 🧠 AI CONFIGURATION (Gemini 3.1 + Grounding + High Thinking)
# ==========================================
if API_KEY:
    genai.configure(api_key=API_KEY)
    
    # 1. Using Gemini 3.1 
    # 2. Enabling Google Search Grounding natively via the tools parameter
    model = genai.GenerativeModel(
        model_name='gemini-3.1-pro', # Upgraded model
        tools=[{"google_search": {}}], # Enables live web search grounding
        system_instruction=(
            "You are an elite, autonomous SRE/DevOps Intelligence Agent. "
            "THINKING LEVEL: HIGH. You must deeply analyze job descriptions, cross-reference "
            "NVIDIA-specific technologies via Google Search if context is missing, and deduce "
            "the precise engineering requirements. Do not output generic summaries. Be hyper-specific."
        )
    )
else:
    print("⚠️ GEMINI_API_KEY not found! AI enrichment will fail.")
    exit(1)

def clean_html(raw_html):
    """Removes HTML tags from the Workday job description"""
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html)

def scrape_nvidia_jobs():
    """Fetches jobs and deeply extracts FULL URL CONTEXT for each job"""
    print(f"🔍 Scraping NVIDIA jobs using keywords: '{SEARCH_KEYWORDS}'...")
    
    # Base search API
    search_url = "https://nvidia.wd5.myworkdayjobs.com/wday/cxs/nvidia/NVIDIAExternalCareerSite/jobs"
    payload = {
        "appliedFacets": {"locationCountry":["bc33aa3152ec42d4995f4791a106ed09"]}, # India
        "limit": 10, # Kept smaller to allow deep context processing per run
        "offset": 0,
        "searchText": SEARCH_KEYWORDS
    }
    
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    response = requests.post(search_url, json=payload, headers=headers)
    
    if response.status_code != 200:
        print("Failed to fetch job list.")
        return[]
    
    jobs_data = response.json().get("jobPostings", [])
    extracted_jobs =[]
    
    # DEEP EXTRACTION: Go into every single job URL to get the full description
    for j in jobs_data:
        external_path = j.get('externalPath')
        public_link = f"https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite{external_path}"
        detail_api_url = f"https://nvidia.wd5.myworkdayjobs.com/wday/cxs/nvidia/NVIDIAExternalCareerSite{external_path}"
        
        print(f"   ↳ Fetching deep URL context for: {j.get('title')}")
        detail_resp = requests.get(detail_api_url, headers={"Accept": "application/json"})
        
        full_description = ""
        if detail_resp.status_code == 200:
            raw_html = detail_resp.json().get('jobPostingInfo', {}).get('jobDescription', '')
            full_description = clean_html(raw_html)

        extracted_jobs.append({
            "title": j.get("title", ""),
            "link": public_link,
            "location": j.get("locationsText", "India"),
            "date_posted": j.get("postedOn", TODAY),
            "full_description": full_description # Full URL context added!
        })
        
    return extracted_jobs

def enrich_with_ai(raw_jobs):
    """Uses Gemini 3.1 High Thinking to extract detailed skills from full descriptions"""
    if not raw_jobs:
        return[]
        
    print(f"🧠 Processing {len(raw_jobs)} jobs with Gemini 3.1 (High Thinking & Search Grounding)...")
    
    prompt = f"""
    Analyze the following highly detailed NVIDIA job descriptions. 
    Use Google Search to ground your understanding of NVIDIA specific tools mentioned in the text (like DGX, Omniverse, NeMo, etc).
    
    RAW JOB DATA:
    {json.dumps(raw_jobs, indent=2)}
    
    Format the output as a STRICT JSON array of objects.
    EACH object must follow this exact structure:
    {{
      "id": "generate_short_hash",
      "title": "Job Title",
      "level": "junior | mid | senior | manager",
      "category": "DevOps | SRE | Platform | MLOps",
      "location": "Location",
      "link": "URL",
      "date_scraped": "{TODAY}",
      "skills":[
        {{
           "name": "Skill Name (e.g., Kubernetes)",
           "description": "Deep, context-aware description of WHY this skill is used here. (e.g., 'Required to manage stateful GPU orchestration across on-prem DGX clusters.')"
        }}
      ],
      "inferred_domain": "e.g., AI Infrastructure, GPU Cloud, Core Systems"
    }}
    
    Output ONLY valid JSON. No markdown blocks.
    """
    
    # Using high thinking implies letting the model output its reasoning. 
    # To force valid JSON while still "thinking", Gemini 3+ handles complex instruction tracking flawlessly.
    generation_config = genai.types.GenerationConfig(
        temperature=0.2, # Low temperature for logical, analytical extraction
        response_mime_type="application/json", # Forces strict JSON generation
    )
    
    response = model.generate_content(prompt, generation_config=generation_config)
    
    try:
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception as e:
        print("Failed to parse AI response. Raw output:", response.text)
        print("Error:", e)
        return[]

def update_reports(enriched_jobs):
    print("📝 Writing intelligent data to repository...")
    
    # 1. Save Daily Snapshot
    with open(f"jobs/{TODAY}.json", "w") as f:
        json.dump(enriched_jobs, f, indent=2)
        
    # 2. Update Master List
    all_jobs_path = "aggregated/all_jobs.json"
    existing_jobs =[]
    if os.path.exists(all_jobs_path):
        with open(all_jobs_path, "r") as f:
            existing_jobs = json.load(f)
            
    seen_links = {j["link"] for j in existing_jobs}
    for job in enriched_jobs:
        if job["link"] not in seen_links:
            job["first_seen"] = TODAY
            job["last_seen"] = TODAY
            existing_jobs.append(job)
            seen_links.add(job["link"])
        else:
            for ej in existing_jobs:
                if ej["link"] == job["link"]:
                    ej["last_seen"] = TODAY
                    ej["skills"] = job["skills"] 
                    
    with open(all_jobs_path, "w") as f:
        json.dump(existing_jobs, f, indent=2)

    # 3. Create High-Level Markdown Table
    md_content = f"# NVIDIA SRE & DevOps Tracker (India)\n*Powered by Gemini 3.1 Pro AI Intelligence*\n*Last Updated: {TODAY}*\n\n"
    md_content += "| Title | Level | Core Skills | Location | Link |\n"
    md_content += "|---|---|---|---|---|\n"
    
    for j in existing_jobs:
        skill_names = [skill["name"] for skill in j.get("skills", [])][:4] 
        skills_preview = ", ".join(skill_names)
        md_content += f"| {j.get('title')} | {j.get('level')} | **{skills_preview}** | {j.get('location')} | [Apply]({j.get('link')}) |\n"
        
    with open("reports/jobs_table.md", "w") as f:
        f.write(md_content)

if __name__ == "__main__":
    raw_jobs = scrape_nvidia_jobs()
    if raw_jobs:
        enriched_jobs = enrich_with_ai(raw_jobs)
        if enriched_jobs:
            update_reports(enriched_jobs)
            print("✅ Autonomous execution complete. Ready to push to GitHub!")
        else:
            print("⚠️ AI failed to return valid data.")
    else:
        print("⏭️ No jobs found today.")
