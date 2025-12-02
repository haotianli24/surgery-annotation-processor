import os
import time
import json
import argparse
import urllib.parse
import zipfile
import requests
from typing import List, Dict, Any, Optional

# ---- DEFAULTS (overridable by CLI) ----
CVAT_URL       = "https://voxel51-cvat.med.umich.edu"   # backend URL, NO trailing slash
#CVAT_URL       = "http://172.17.2.33:8082"   # backend URL, NO trailing slash
USER           = "filmp"
PASS           = "filmpellakos"
TASK_NAMES     = [
    "001_017_SF_3",
    "001_017_SF_10"
]            # multiple names supported
OUT_DIR        = "./cvat_exports"
VERIFY_SSL     = False
EXPORT_FORMAT  = "CVAT for video 1.1"       # change if needed
SAVE_IMAGES    = False                       # include images in export
SKIP_IF_ZIP_EXISTS = True                    # skip if zip already present
MAX_POLLS      = 60                          # 2 min at 2s per poll
POLL_SLEEP_S   = 2
PAGE_SIZE      = 100

def normalize_url_scheme(url: str, base_url: str) -> str:
    """Ensure URL uses same scheme (http/https) as base_url"""
    if url.startswith("/"):
        return ujoin(base_url, url)
    
    base_scheme = "https://" if base_url.startswith("https://") else "http://"
    wrong_scheme = "http://" if base_scheme == "https://" else "https://"
    
    if url.startswith(wrong_scheme):
        return url.replace(wrong_scheme, base_scheme, 1)
    
    return url

def ujoin(base: str, path: str) -> str:
    return urllib.parse.urljoin(base.rstrip("/") + "/", path.lstrip("/"))

def ok(r: requests.Response, expect_json: bool = False):
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"{r.request.method} {r.url} -> {r.status_code}\n{(r.text or '')[:500]}")
    return r.json() if expect_json else r

def get_all_tasks(session: requests.Session, base_url: str, verify: bool = True) -> List[Dict[str, Any]]:
    """Fetch all tasks from CVAT API across all pages"""
    all_tasks = []
    page = 1
    while True:
        print(f"Fetching tasks page {page}...")
        params = {"page": page, "page_size": PAGE_SIZE}
        response = ok(session.get(ujoin(base_url, "/api/tasks"), params=params, verify=verify), True)
        tasks = response.get("results", [])
        all_tasks.extend(tasks)
        if not response.get("next"):
            break
        page += 1
    print(f"Found {len(all_tasks)} total tasks across {page} pages")
    return all_tasks

def ensure_export_format(session: requests.Session, base_url: str, desired: str, verify: bool = True) -> str:
    print("Fetching available export formats...")
    formats_resp = ok(session.get(ujoin(base_url, "/api/server/annotation/formats"), verify=verify), True)
    exporters = formats_resp.get("exporters", [])
    available = [f["name"] for f in exporters if f.get("name")]
    print("Available exporters:")
    for fmt in available:
        print(f"  - {fmt}")
    if desired not in available:
        print(f"Warning: Format '{desired}' not found. Choosing a CVAT-* format if present; else first available.")
        cvat_formats = [f for f in available if "CVAT" in f]
        chosen = cvat_formats[0] if cvat_formats else (available[0] if available else desired)
        print(f"Using format: {chosen}")
        return chosen
    return desired

def find_task_by_name(tasks: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    match = [t for t in tasks if t.get("name") == name]
    if match:
        return match[0]
    # helpful diagnostics
    print(f"Task '{name}' not found among {len(tasks)} tasks.")
    sample = [t.get("name") for t in tasks[:20]]
    print(f"First 20 available tasks: {sample}")
    partial = [t.get("name") for t in tasks if name.lower() in str(t.get("name","")).lower()]
    if partial:
        print(f"Partial matches: {partial[:10]}")
    return None

def start_export(session: requests.Session, base_url: str, task_id: int, export_format: str,
                 save_images: bool, verify: bool = True) -> Dict[str, Any]:
    export_url = ujoin(base_url, f"/api/tasks/{task_id}/dataset/export")
    params = {"format": export_format, "save_images": str(save_images).lower()}
    print(f"POST {export_url} params={params}")
    r = session.post(export_url, params=params, verify=verify)
    print(f"Export initiation status: {r.status_code}")
    if r.status_code != 202:
        print(f"Export failed to start: {r.text[:500]}")
        return {"error": True, "status_code": r.status_code, "text": r.text}
    data = r.json()
    print(f"Export started. rq_id={data.get('rq_id')} result_url={data.get('result_url')}")
    return data

def poll_request(session: requests.Session, base_url: str, rq_id: str, verify: bool = True,
                 max_polls: int = MAX_POLLS, sleep_s: int = POLL_SLEEP_S) -> Dict[str, Any]:
    status_url = ujoin(base_url, f"/api/requests/{rq_id}")
    print(f"Polling status at: {status_url}")
    for poll in range(1, max_polls + 1):
        print(f"  Poll {poll}/{max_polls}...")
        resp = session.get(status_url, verify=verify)
        if resp.status_code != 200:
            print(f"Status check failed: {resp.status_code} {resp.text[:200]}")
            break
        data = resp.json()
        status = data.get("status")
        progress = data.get("progress", 0)
        print(f"    Status: {status}, Progress: {progress}%")
        if status == "finished":
            print("Export completed successfully!")
            return {"ok": True, "data": data}
        if status == "failed":
            print(f"Export failed. stderr:\n{data.get('stderr','<no details>')}")
            return {"ok": False, "data": data}
        time.sleep(sleep_s)
    print("Export timed out or status unknown")
    return {"ok": False, "data": {}}

def download_to(session: requests.Session, url: str, dest_path: str, verify: bool = True) -> bool:
    print(f"Downloading: {url}")
    with session.get(url, stream=True, verify=verify) as r:
        if r.status_code != 200:
            print(f"Download failed: {r.status_code} {r.text[:200]}")
            return False
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"Saved: {dest_path} ({os.path.getsize(dest_path)} bytes)")
    return True

def try_extract(zip_path: str, dest_dir: str):
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest_dir)
        print(f"Extracted to: {dest_dir}")
    except zipfile.BadZipFile:
        print("Note: Downloaded file is not a zip archive")
    except Exception as e:
        print(f"Could not extract: {e}")

def checker_fallbacks(session: requests.Session, base_url: str, task_id: int, task_name: str,
                      export_format: str, out_dir: str, verify: bool = True) -> bool:
    dest_dir = os.path.join(out_dir, task_name)
    os.makedirs(dest_dir, exist_ok=True)
    fallback_zip = os.path.join(dest_dir, f"{task_name}_from_fallback.zip")

    # Direct GET
    try:
        direct = session.get(
            ujoin(base_url, f"/api/tasks/{task_id}/dataset/export"),
            params={"format": export_format}, verify=verify, stream=True
        )
        print(f"Direct export GET status: {direct.status_code}")
        if direct.status_code == 200:
            with open(fallback_zip, "wb") as f:
                for chunk in direct.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            print(f"Fallback direct download saved: {fallback_zip}")
            try_extract(fallback_zip, dest_dir)
            return True
    except Exception as e:
        print(f"Direct GET fallback error: {e}")

    # Requests scan
    try:
        reqs = ok(session.get(ujoin(base_url, "/api/requests"), verify=verify), True)
        results = reqs.get("results", [])
        print(f"Found {len(results)} recent requests; scanning for finished exports...")
        for req in results:
            if req.get("status") == "finished" and req.get("result_url"):
                result_url = req.get("result_url")
                if result_url.startswith("/"):
                    result_url = ujoin(base_url, result_url)
                if download_to(session, result_url, fallback_zip, verify=verify):
                    try_extract(fallback_zip, dest_dir)
                    return True
    except Exception as e:
        print(f"Requests-scan fallback error: {e}")

    return False

def process_task(session: requests.Session, base_url: str, task: Dict[str, Any],
                 export_format: str, save_images: bool, out_dir: str,
                 verify: bool = True, skip_if_zip_exists: bool = True):
    task_name = task.get("name")
    task_id = task.get("id")
    print(f"\n===== Task: '{task_name}' (ID {task_id}) =====")

    dest_dir = os.path.join(out_dir, task_name)
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, f"{task_name}_export.zip")

    if skip_if_zip_exists and os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
        print(f"Skipping export: {zip_path} already exists")
        return

    data = start_export(session, base_url, task_id, export_format, save_images, verify=verify)
    if data.get("error"):
        print("Attempting fallback checker routes since export start did not return 202...")
        if checker_fallbacks(session, base_url, task_id, task_name, export_format, out_dir, verify):
            return
        else:
            print("No fallback succeeded.")
            return

    rq_id = data.get("rq_id")
    result_url = data.get("result_url")

    if not rq_id:
        print("No rq_id returned; trying fallback checker...")
        checker_fallbacks(session, base_url, task_id, task_name, export_format, out_dir, verify)
        return

    poll_res = poll_request(session, base_url, rq_id, verify=verify)
    if not poll_res.get("ok"):
        print("Polling did not succeed; trying fallback checker...")
        checker_fallbacks(session, base_url, task_id, task_name, export_format, out_dir, verify)
        return

    if not result_url:
        result_url = poll_res["data"].get("result_url")

    if result_url:
        # if result_url.startswith("/"):
        #     result_url = ujoin(base_url, result_url)
        result_url = normalize_url_scheme(result_url, base_url)
    
        if download_to(session, result_url, zip_path, verify=verify):
            try_extract(zip_path, dest_dir)
        else:
            print("Primary download failed; trying fallback checker...")
            checker_fallbacks(session, base_url, task_id, task_name, export_format, out_dir, verify)
    else:
        print("No result_url provided; trying fallback checker...")
        checker_fallbacks(session, base_url, task_id, task_name, export_format, out_dir, verify)

def main():
    parser = argparse.ArgumentParser(description="Export multiple CVAT tasks by name (2.39.1).")
    parser.add_argument("--cvat-url", default=CVAT_URL)
    parser.add_argument("--user", default=USER)
    parser.add_argument("--password", default=PASS)
    parser.add_argument("--task-names", default=",".join(TASK_NAMES),
                        help="Comma-separated task names, e.g. '001_023_SF_3,003_015_OT_3'")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--verify-ssl", action="store_true", default=VERIFY_SSL)
    parser.add_argument("--no-verify-ssl", action="store_true", help="Disable SSL verification")
    parser.add_argument("--export-format", default=EXPORT_FORMAT)
    parser.add_argument("--save-images", action="store_true", default=SAVE_IMAGES)
    parser.add_argument("--no-skip-if-zip-exists", action="store_true",
                        help="If set, will re-export even if zip exists")
    args = parser.parse_args()

    base_url = args.cvat_url
    verify = False if args.no_verify_ssl else args.verify_ssl
    out_dir = args.out_dir
    export_format = args.export_format
    save_images = args.save_images
    skip_if_zip_exists = not args.no_skip_if_zip_exists

    task_names = [t.strip() for t in args.task_names.split(",") if t.strip()]
    if not task_names:
        print("No task names provided.")
        return

    os.makedirs(out_dir, exist_ok=True)

    s = requests.Session()
    s.auth = (args.user, args.password)

    about = ok(s.get(ujoin(base_url, "/api/server/about"), verify=verify), True)
    print("CVAT version:", about.get("version"))
    me = ok(s.get(ujoin(base_url, "/api/users/self"), verify=verify), True)
    print("Logged in as:", me.get("username"))

    export_format = ensure_export_format(s, base_url, export_format, verify=verify)

    print("Fetching all tasks (with pagination)...")
    tasks = get_all_tasks(s, base_url, verify=verify)

    for name in task_names:
        task = find_task_by_name(tasks, name)
        if not task:
            print(f"Skipping '{name}' (not found).")
            continue
        process_task(
            s, base_url, task,
            export_format=export_format,
            save_images=save_images,
            out_dir=out_dir,
            verify=verify,
            skip_if_zip_exists=skip_if_zip_exists
        )

    print("\n=== Directory Summary ===")
    if os.path.exists(out_dir):
        for entry in sorted(os.listdir(out_dir)):
            p = os.path.join(out_dir, entry)
            if os.path.isdir(p):
                print(f"[DIR] {p}")
                for f in sorted(os.listdir(p)):
                    fp = os.path.join(p, f)
                    if os.path.isfile(fp):
                        print(f"  - {f} ({os.path.getsize(fp)} bytes)")
    else:
        print(f"{out_dir} does not exist.")

if __name__ == "__main__":
    main()