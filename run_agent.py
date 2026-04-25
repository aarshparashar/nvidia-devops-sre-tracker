import os
import json
import datetime
import time
import requests
from bs4 import BeautifulSoup
import re
import concurrent.futures
from urllib.parse import urlparse
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
# If true, accept some live Workday pages that omit JobPosting in HTML (may re-admit bad links — use for debugging).
JOB_VERIFY_RELAXED = os.environ.get("JOB_VERIFY_RELAXED", "").lower() in ("1", "true", "yes")
# If set, do not call the API a second time when the first response parses as an empty job list [].
GEMINI_SKIP_EMPTY_RETRY = os.environ.get("GEMINI_SKIP_EMPTY_RETRY", "").lower() in ("1", "true", "yes")

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


def _normalize_job_url(link: str) -> str:
    if not link or not isinstance(link, str):
        return ""
    link = link.strip()
    if link.startswith("//"):
        link = "https:" + link
    return link


def _jsonld_type_is_jobposting(node: dict) -> bool:
    t = node.get("@type")
    if t == "JobPosting":
        return True
    if isinstance(t, list):
        return "JobPosting" in t
    return False


def _walk_jsonld_for_jobposting(data) -> bool:
    if isinstance(data, dict):
        if _jsonld_type_is_jobposting(data):
            return True
        graph = data.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                if _walk_jsonld_for_jobposting(item):
                    return True
        for v in data.values():
            if isinstance(v, (dict, list)) and _walk_jsonld_for_jobposting(v):
                return True
    elif isinstance(data, list):
        for item in data:
            if _walk_jsonld_for_jobposting(item):
                return True
    return False


def _html_has_jobposting_schema(text: str) -> bool:
    """Detect schema.org JobPosting in page (substring or parsed ld+json)."""
    if "JobPosting" in text or "jobPosting" in text:
        return True
    try:
        soup = BeautifulSoup(text, "html.parser")
        for script in soup.find_all("script", {"type": "application/ld+json"}):
            if not script.string or not script.string.strip():
                continue
            try:
                data = json.loads(script.string)
            except json.JSONDecodeError:
                continue
            if _walk_jsonld_for_jobposting(data):
                return True
    except Exception:
        pass
    return False


def _workday_detail_page_heuristic(url: str, text: str) -> bool:
    """
    GitHub Actions often gets a large Workday shell without ld+json JobPosting.
    Accept only canonical job-detail URLs plus multiple in-page signals.
    """
    if not re.search(r"_JR\d+", url, re.I):
        return False
    path = (urlparse(url).path or "").lower()
    if "/job/" not in path:
        return False
    if len(text) < 4500:
        return False
    low = text.lower()
    if "myworkdayjobs" not in low:
        return False
    kws = (
        "requisitionid",
        "jobpostinginfo",
        "jobdetails",
        "careersection",
        "timeposted",
        "bulletfields",
        "jobdescription",
        "similarjobs",
        "jobreqid",
    )
    return sum(1 for k in kws if k in low) >= 3


def verify_nvidia_job_url_detail(url: str, timeout: int = 12) -> tuple[bool, str]:
    """Return (ok, reason). reason is machine-readable for logs and audit JSON."""
    url = _normalize_job_url(url)
    if not url:
        return False, "empty_or_missing_link"
    if not url.startswith("https://"):
        return False, "not_https"
    host = urlparse(url).netloc.lower()
    if host != "nvidia.wd5.myworkdayjobs.com":
        return False, f"wrong_host:{host or 'none'}"

    last_err = None
    for attempt, tmo in enumerate((timeout, timeout + 6), start=1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=tmo, allow_redirects=True)
            if r.status_code != 200:
                return False, f"http_{r.status_code}"
            text = r.text
            if _html_has_jobposting_schema(text):
                return True, "ok_jobposting_in_html"
            if JOB_VERIFY_RELAXED and len(text) > 8000 and "myworkdayjobs" in text.lower():
                return True, "ok_relaxed_large_workday_page"
            if _workday_detail_page_heuristic(url, text):
                return True, "ok_workday_detail_heuristic"
            return False, "http_200_but_no_jobposting_schema"
        except requests.RequestException as e:
            last_err = f"request_error:{type(e).__name__}"
            if attempt >= 2:
                return False, last_err
    return False, last_err or "request_failed"


def verify_nvidia_job_url(url: str, timeout: int = 12) -> bool:
    return verify_nvidia_job_url_detail(url, timeout=timeout)[0]


def _verify_one_link(link):
    return verify_nvidia_job_url_detail(link)


def filter_jobs_by_verified_links(jobs: list, *, stage: str = "filter") -> tuple[list, list[dict]]:
    """Return (kept_jobs, rejected_rows) where each rejected row has title, link, reason."""
    if not jobs:
        return [], []
    links = [j.get("link") for j in jobs]
    max_workers = min(8, max(1, len(links)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        outcomes = list(ex.map(_verify_one_link, links))

    kept = []
    rejected = []
    for j, (ok, reason) in zip(jobs, outcomes):
        if ok:
            kept.append(j)
        else:
            rejected.append({
                "title": j.get("title", ""),
                "link": j.get("link", ""),
                "reason": reason,
            })

    dropped = len(rejected)
    if dropped:
        mode = "relaxed" if JOB_VERIFY_RELAXED else "strict"
        print(
            f"   ↳ URL check ({stage}, {mode}): kept {len(kept)}/{len(jobs)}; "
            f"rejected {dropped} (see lines below and jobs/url_verification_audit_{TODAY}.json)."
        )
        for row in rejected:
            t = (row.get("title") or "untitled")[:72]
            u = row.get("link") or ""
            print(f"      — rejected: {t}")
            print(f"        url: {u}")
            print(f"        reason: {row.get('reason')}")

    return kept, rejected


def scrape_nvidia_jobs():
    """Scrapes jobs by falling back to Google Search due to Workday's strict bot protections on their own site."""
    print("🔍 Searching for NVIDIA India DevOps/SRE jobs via external index...")
    return []


def _extract_model_text(response) -> str:
    """Best-effort text extraction; `response.text` is sometimes empty for tool-using / blocked responses."""
    t = getattr(response, "text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()
    out = []
    for c in getattr(response, "candidates", None) or []:
        content = getattr(c, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            continue
        for p in parts:
            pt = getattr(p, "text", None)
            if isinstance(pt, str) and pt:
                out.append(pt)
    return "\n".join(out).strip()


def _parse_jobs_json(raw: str) -> list:
    """Parse model output into a list of job dicts; tolerate fences and trailing prose."""
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty_model_output")
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        lo, hi = raw.find("["), raw.rfind("]")
        if lo == -1 or hi <= lo:
            raise
        data = json.loads(raw[lo : hi + 1])
    if not isinstance(data, list):
        raise ValueError("model_json_not_a_list")
    return data


def _log_generate_response_debug(response) -> None:
    pf = getattr(response, "prompt_feedback", None)
    if pf is not None:
        print(f"   ↳ prompt_feedback: {pf}")
    cands = getattr(response, "candidates", None) or []
    if not cands:
        print("   ↳ candidates: (none)")
        return
    c0 = cands[0]
    fr = getattr(c0, "finish_reason", None)
    print(f"   ↳ first_candidate.finish_reason: {fr!r}")


def enrich_with_ai() -> tuple[list, str]:
    """Call Gemini + search. Returns (jobs, status) where status is ok | empty_list | empty_response | parse_error."""
    print(
        f"🧠 Using {GEMINI_MODEL} with Live Search Grounding to find and analyze today's jobs..."
    )

    base_prompt = f"""
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
      "link": "ONLY a real, working https URL on nvidia.wd5.myworkdayjobs.com that you have seen in search results — never invent or guess requisition IDs",
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

    retry_nudge = """
    FOLLOW-UP (only if your previous answer was []): Run additional searches, e.g.
    site:nvidia.wd5.myworkdayjobs.com India (DevOps OR \"Site Reliability\" OR SRE OR Platform OR MLOps).
    Include every open India role you can verify with a real nvidia.wd5.myworkdayjobs.com URL from search snippets.
    Output [] only if you have confirmed there are zero matching open roles after those searches.
    """

    prompts: list[str] = [base_prompt]
    if not GEMINI_SKIP_EMPTY_RETRY:
        prompts.append(base_prompt + retry_nudge)

    config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
        system_instruction="THINKING LEVEL: HIGH. You are a precise data extraction bot.",
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )

    for attempt_idx, prompt in enumerate(prompts):
        response = None
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config
            )
            raw = _extract_model_text(response)
            if not raw:
                _log_generate_response_debug(response)
                print("Failed to parse AI response. Error: empty_model_output")
                return [], "empty_response"
            jobs = _parse_jobs_json(raw)
            if jobs:
                if attempt_idx > 0:
                    print(f"✅ AI returned {len(jobs)} job(s) after one retry.")
                else:
                    print(f"✅ AI returned {len(jobs)} job(s).")
                return jobs, "ok"
            # Parsed OK but empty list
            if attempt_idx < len(prompts) - 1:
                print(
                    "⚠️ AI returned a valid empty JSON array []. Retrying once with stronger search instructions "
                    "(set GEMINI_SKIP_EMPTY_RETRY=1 to disable this extra API call)."
                )
                time.sleep(2)
                continue
            print(
                "ℹ️ AI still returned [] after retry — treating as 'no matching listings found' for this run "
                "(search grounding can be flaky; next scheduled run may differ)."
            )
            return [], "empty_list"
        except Exception as e:
            if response is not None:
                try:
                    _log_generate_response_debug(response)
                except Exception:
                    pass
            print("Failed to parse AI response. Error:", e)
            return [], "parse_error"

    return [], "parse_error"


def _write_url_verification_audit(
    after_ai: list[dict],
    for_markdown: list[dict],
    *,
    extra: dict | None = None,
) -> None:
    path = f"jobs/url_verification_audit_{TODAY}.json"
    payload = {
        "date": TODAY,
        "strict_mode": not JOB_VERIFY_RELAXED,
        "after_ai": {"rejected": after_ai},
        "markdown_table": {"rejected": for_markdown},
        "how_to_read": (
            "Rejected rows failed automated checks: wrong host, non-200 HTTP, or no JobPosting JSON-LD in HTML. "
            "Open each link in a browser; if it is a real job, the heuristic may be too strict — set JOB_VERIFY_RELAXED=1 "
            "on one workflow run to compare, or relax verify_nvidia_job_url_detail in run_agent.py."
        ),
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _finalize_markdown_and_audit(
    existing_jobs: list,
    *,
    ai_url_rejections: list[dict] | None = None,
    run_note: str | None = None,
    audit_extra: dict | None = None,
) -> None:
    md_content = (
        f"# NVIDIA SRE & DevOps Tracker (India)\n"
        f"*Powered by Gemini 3.1 Pro Preview + Live Search Grounding*\n"
        f"*Last Updated: {TODAY}*\n"
    )
    if run_note:
        md_content += f"{run_note}\n"
    md_content += "\n| Title | Level | Core Skills | Location | Link |\n"
    md_content += "|---|---|---|---|---|\n"

    rows_for_md, rejected_md = filter_jobs_by_verified_links(
        existing_jobs, stage="markdown_table"
    )
    if len(rows_for_md) < len(existing_jobs):
        print(
            f"   ↳ jobs_table.md: {len(rows_for_md)}/{len(existing_jobs)} rows have verified Apply URLs."
        )
    rows_for_md.sort(key=lambda x: x.get("last_seen", ""), reverse=True)

    for j in rows_for_md:
        skill_names = [skill["name"] for skill in j.get("skills", [])][:4]
        skills_preview = ", ".join(skill_names)
        md_content += f"| {j.get('title', 'N/A')} | {j.get('level', 'N/A')} | **{skills_preview}** | {j.get('location', 'India')} | [Apply]({j.get('link', '#')}) |\n"

    with open("jobs_table.md", "w") as f:
        f.write(md_content)

    _write_url_verification_audit(
        ai_url_rejections or [],
        rejected_md,
        extra=audit_extra,
    )


def refresh_artifacts_after_ai_failure(*, ai_status: str) -> None:
    """Ensure jobs_table.md exists for git; rebuild from aggregate when AI returns no new rows."""
    print(
        "📝 Rebuilding jobs_table.md from saved aggregate (AI returned no job rows this run)..."
    )
    all_jobs_path = "aggregated/all_jobs.json"
    existing_jobs: list = []
    if os.path.exists(all_jobs_path):
        with open(all_jobs_path, "r") as f:
            existing_jobs = json.load(f)

    if ai_status == "empty_list":
        note = (
            "*Run note: The model returned a valid empty list `[]` (no new rows this run). "
            "Table below is from `aggregated/all_jobs.json` when present.*"
            if existing_jobs
            else "*Run note: The model returned `[]` and there is no aggregate file yet.*"
        )
    elif ai_status == "empty_response":
        note = (
            "*Run note: The model returned no text; table from `aggregated/all_jobs.json` only.*"
            if existing_jobs
            else "*Run note: Empty model response and no aggregate file yet.*"
        )
    else:
        note = (
            "*Run note: AI response could not be parsed as JSON; table from `aggregated/all_jobs.json` only.*"
            if existing_jobs
            else "*Run note: AI parse failed and there is no aggregate file yet.*"
        )

    _finalize_markdown_and_audit(
        existing_jobs,
        ai_url_rejections=[],
        run_note=note,
        audit_extra={"ai_outcome": ai_status},
    )


def update_reports(enriched_jobs, *, ai_url_rejections: list[dict] | None = None):
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

    _finalize_markdown_and_audit(existing_jobs, ai_url_rejections=ai_url_rejections)


if __name__ == "__main__":
    raw_jobs, ai_status = enrich_with_ai()
    if not raw_jobs:
        if ai_status == "empty_list":
            print(
                "⏭️ No new job rows from AI this run (empty list). "
                "Refreshing table from aggregate if available."
            )
        elif ai_status == "empty_response":
            print("⏭️ Empty model output. Refreshing table from aggregate if available.")
        else:
            print("⏭️ AI response could not be parsed. Refreshing table from aggregate if available.")
        refresh_artifacts_after_ai_failure(ai_status=ai_status)
    else:
        verified_jobs, rejected_ai = filter_jobs_by_verified_links(
            raw_jobs, stage="after_ai"
        )
        update_reports(verified_jobs, ai_url_rejections=rejected_ai)
        if verified_jobs:
            print("✅ Autonomous execution complete. Ready to push to GitHub!")
        else:
            print(
                "⏭️ No new jobs passed URL verification; aggregated data and jobs_table.md were still refreshed using live-link checks."
            )
