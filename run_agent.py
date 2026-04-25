import os
import json
import datetime
import requests
from bs4 import BeautifulSoup
import re
import concurrent.futures
from google import genai
from google.genai import types

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# We use the public search URL with URL parameters instead of hidden JSON APIs
SEARCH_URL = "https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite?q=DevOps+SRE+Reliability+Platform+MLOps+Infrastructure"
TODAY = datetime.datetime.now().strftime("%Y-%m-%d")
API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

for folder in ["jobs", "aggregated"]:
    os.makedirs(folder, exist_ok=True)

if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    print("⚠️ GEMINI_API_KEY not found! AI enrichment will fail.")
    exit(1)


def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html)


def fetch_deep_job_context(job_link):
    """Fetches full description by directly loading the job's public HTML page"""
    print(f"   ↳ Fetching deep context from: {job_link}")

    try:
        resp = requests.get(job_link, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')

            title_text = soup.title.string if soup.title else "NVIDIA Job"
            title = title_text.split(" | ")[0].strip()

            script_tags = soup.find_all("script", {"type": "application/ld+json"})
            full_description = ""
            location = "India"

            for script in script_tags:
                try:
                    data = json.loads(script.string)
                    if "@type" in data and data["@type"] == "JobPosting":
                        full_description = clean_html(data.get("description", ""))

                        loc_data = data.get("jobLocation", {})
                        if isinstance(loc_data, dict):
                            address = loc_data.get("address", {})
                            country = address.get("addressCountry", "")
                            region = address.get("addressRegion", "")
                            city = address.get("addressLocality", "")
                            location = f"{city}, {region}, {country}".strip(", ")
                        elif isinstance(loc_data, list):
                            location = "Multiple Locations"
                except json.JSONDecodeError:
                    pass

            return {
                "title": title,
                "link": job_link,
                "location": location,
                "date_posted": TODAY,
                "full_description": full_description
            }
    except Exception as e:
        print(f"Failed to load {job_link}: {e}")

    return None


def scrape_nvidia_jobs():
    """Scrapes jobs by falling back to Google Search due to Workday's strict bot protections on their own site."""
    print("🔍 Searching for NVIDIA India DevOps/SRE jobs via external index...")
    return []


def enrich_with_ai():
    """Uses Gemini 3.1 Pro Preview with Google Search to dynamically FIND and ENRICH the jobs in one step!"""
    print(
        f"🧠 Using {GEMINI_MODEL} with Live Search Grounding to find and analyze today's jobs..."
    )

    prompt = f"""
    You are an elite, autonomous SRE/DevOps Intelligence Agent.

    TASK: Use Google Search to find the latest DevOps, Site Reliability Engineering, and Platform Engineering job postings at NVIDIA specifically located in INDIA.
    Look for roles posted recently.

    For every real, currently open NVIDIA job you find in India that matches these categories, you must analyze its requirements based on your search context.

    Format the output as a STRICT JSON array of objects.
    EACH object must follow this exact structure:
    {{
      "id": "generate_short_hash",
      "title": "Exact Job Title",
      "level": "junior | mid | senior | manager",
      "category": "DevOps | SRE | Platform | MLOps",
      "location": "City, India",
      "link": "URL to the job posting (must be a real nvidia.wd5 URL if possible)",
      "date_scraped": "{TODAY}",
      "skills":[
        {{
           "name": "Skill Name (e.g., Kubernetes)",
           "description": "Deep, context-aware description of WHY this skill is used here."
        }}
      ],
      "inferred_domain": "e.g., AI Infrastructure, GPU Cloud, Core Systems"
    }}

    Output ONLY valid JSON. No markdown blocks. If you cannot find any recent jobs, output an empty array[].
    """

    config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
        system_instruction="THINKING LEVEL: HIGH. You are a precise data extraction bot.",
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config
        )
        raw = (response.text or "").replace("```json", "").replace("```", "").strip()
        jobs = json.loads(raw)
        print(f"✅ AI successfully found and processed {len(jobs)} jobs!")
        return jobs
    except Exception as e:
        print("Failed to parse AI response. Error:", e)
        return []


def update_reports(enriched_jobs):
    print("📝 Writing intelligent data to repository...")

    with open(f"jobs/{TODAY}.json", "w") as f:
        json.dump(enriched_jobs, f, indent=2)

    all_jobs_path = "aggregated/all_jobs.json"
    existing_jobs = []
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

    md_content = f"# NVIDIA SRE & DevOps Tracker (India)\n*Powered by Gemini 3.1 Pro Preview + Live Search Grounding*\n*Last Updated: {TODAY}*\n\n"
    md_content += "| Title | Level | Core Skills | Location | Link |\n"
    md_content += "|---|---|---|---|---|\n"

    existing_jobs.sort(key=lambda x: x.get("last_seen", ""), reverse=True)

    for j in existing_jobs:
        skill_names = [skill["name"] for skill in j.get("skills", [])][:4]
        skills_preview = ", ".join(skill_names)
        md_content += f"| {j.get('title', 'N/A')} | {j.get('level', 'N/A')} | **{skills_preview}** | {j.get('location', 'India')} | [Apply]({j.get('link', '#')}) |\n"

    with open("jobs_table.md", "w") as f:
        f.write(md_content)


if __name__ == "__main__":
    enriched_jobs = enrich_with_ai()
    if enriched_jobs:
        update_reports(enriched_jobs)
        print("✅ Autonomous execution complete. Ready to push to GitHub!")
    else:
        print("⏭️ No jobs found today or AI search failed.")
