import os
import re
import requests
import sys
from urllib.parse import urlparse
sys.path.append("../")
import llm.ask_llm
import json


_URL_BLACKLIST_PATTERNS = (
    re.compile(r'(?i)(?:^|/)(?:issues|pull|compare)(?:/|$)'),
    re.compile(r'(?i)(?:^|/)(?:forum|forums|bugzilla|trac)(?:[/.]|$)'),
    re.compile(r'(?i)(?:^|/)(?:license|licenses|licensing|copying)(?:[/.]|$)'),
    re.compile(r'(?i)\.dtd(?:$|[?#])'),
    re.compile(r'(?i)(?:^|/)(?:xml|xsd|schema|schemas)(?:[/.]|$)'),
)

_URL_BLACKLIST_HOSTS = {
    "schemas.microsoft.com",
    "www.w3.org",
}


async def parse_level2_directory(compile_path):
    result = {"files": [], "directories": []}
    def walk(path, depth):
        for entry in os.scandir(path):
            abs_path = entry.path
            rel = os.path.relpath(abs_path, compile_path)
            # rel = abs_path
            if entry.is_file():
                ext = os.path.splitext(entry.name)[1].lower()
                if ext not in (".cc", ".c", ".c++", ".h", ".cpp", ".hpp", ".cxx", ".hxx", ".hh"):
                    result["files"].append(rel)
                # if any(x in entry.name.lower() for x in ['install', 'readme', 'compile', 'require', 'depend'] ):
                #     result["可疑文件"].append(rel)
            elif entry.is_dir():
                result["directories"].append(rel)
                if depth > 1:
                    walk(abs_path, depth - 1)
    try:
        walk(compile_path, depth=2)
    except Exception as e:
        print(e)
    print(result)
    return result

def extract_http(content):
    # 匹配 http 或 https 开头的 URL，直到遇到空格、引号或结束括号
    pattern = r'https?://[^\s\'")>\]}]+'
    return re.findall(pattern, content, )

def _normalize_link_candidate(link: str) -> str:
    link = str(link or "").strip()
    if not link:
        return ""
    return link.rstrip('\'".,;:!?)]}>')


def _is_blacklisted_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    full = f"{host}{path}"
    if host in _URL_BLACKLIST_HOSTS:
        return True
    return any(pattern.search(full) for pattern in _URL_BLACKLIST_PATTERNS)


def extract_all_links(markdown_text):
    # 提取 []() 和 ![]() 中的链接部分，包括以 / 开头的链接
    pattern_md = r'!?\[.*?\]\(([^)\s]+)\)'
    pattern_plain = r'https?://[^\s)>\]]+'  # 避免末尾包含括号、右方括号等

    links_md = re.findall(pattern_md, markdown_text)
    links_plain = re.findall(pattern_plain, markdown_text)

    ordered = []
    seen = set()
    for raw_link in links_md + links_plain:
        link = _normalize_link_candidate(raw_link)
        if not link or link.startswith(("#", "mailto:", "javascript:")):
            continue
        if link in seen:
            continue
        seen.add(link)
        ordered.append(link)
    return ordered


async def regex_get_urls(filepaths, repopath):
    url_collects = []
    filepath_collects = []
    for relfilepath in filepaths:
        filepath = os.path.join(repopath, relfilepath)
        if not os.path.isfile(filepath):
            continue
        try:
            content = open(filepath, "r", encoding='utf-8', errors='ignore').read()
        except OSError:
            continue
        extract_links = extract_all_links(content)
        for link in extract_links:
            if link.startswith("http"):
                if _is_blacklisted_url(link):
                    continue
                url_collects.append(link)
                continue
            tmppath = link
            if link.startswith("/"):
                tmppath = tmppath[1:]
            filepath_collects.append(os.path.join(repopath, tmppath))
    return url_collects, filepath_collects


async def Fetch_URL_Content(url, timeout=10):
    from bs4 import BeautifulSoup

    from fake_useragent import UserAgent
    ua = UserAgent()
    headers = {
        "User-Agent": ua.random,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    proxies = {
        "http": os.environ.get("HTTP_PROXY"),
        "https": os.environ.get("HTTPS_PROXY")
    }

    try:
        response = requests.get(url, headers=headers, proxies=proxies,  timeout=timeout)
        if response.ok:
            soup = BeautifulSoup(response.text, "html.parser")
            texts = soup.get_text(separator=" #newline ",strip=True)
            return (True, texts)
        else:
            return (False, None)
    except requests.exceptions.RequestException as e:
        return (False, None)
    


# TODO: URL summary with recursion
import asyncio

async def summarize_urls(repopath, initial_urls, prompt_sys, prompt_usr):
    left_urls = list(initial_urls)
    depth = 0
    url_contents = {}
    visited_urls = set()
    max_depth = 2  # 可调节的递归深度限制
    print("Initial URLs:", left_urls)
    while left_urls and depth < max_depth:
        print(f"[Depth {depth}] Processing {len(left_urls)} URLs...")
        next_urls = []
        maxurl = 2
        for url in left_urls:
            if not url.startswith("http") or url in visited_urls:
                continue
            visited_urls.add(url)
            if maxurl <= 0:
                print(f"[Depth {depth}] Max URL limit reached, stopping further processing.")
                next_urls = []
                break
            try:
                print(f"[Fetching] {url}")
                success, content = await Fetch_URL_Content(url)
            except Exception as e:
                print(f"[Error] Exception while fetching {url}: {e}")
                continue
            if success:
                attempt = 0
                while attempt < 2:
                    try:
                        url_summary = ask_llm.ask_ds(
                            prompt_sys,
                            prompt_usr.format(content=content)
                        )
                        
                        print("done: ", url, url_summary)
                        if url_summary and isinstance(url_summary, str):
                            url_summary = re.sub(r"```[a-zA-Z]*\n?", "", url_summary).strip()
                        if url_summary and isinstance(url_summary, list):
                            url_summary = url_summary[0]
                        # if isinstance(llm_related, list):
                        #     llm_related = llm_related[0]
                        # if isinstance(url_summary, str):
                        url_summary = json.loads(url_summary)
                        if all(key in url_summary for key in ["summary", "urls"]):
                            url_contents[url] = url_summary["summary"]
                            new_urls = [
                                new_url for new_url in url_summary["urls"]
                                if new_url.startswith("http") and new_url not in visited_urls
                            ]
                            next_urls.extend(new_urls)
                            break
                    except Exception as e:
                        attempt += 1 
                        print("error", e)
                        continue
                print("done for ", url)
                maxurl -= 1
            else:
                print(f"[Fetch Failed] {url}")

        left_urls = next_urls
        depth += 1
    print("Visted URLs:", visited_urls)
    return url_contents


if __name__ == "__main__":
    # Example usage 
    # compile_path = "/path/to/your/repo"
    # asyncio.run(parse_level2_directory(compile_path))
    # You can also test the URL fetching and summarization here
    urls = ["https://github.com/heimdal/heimdal/wiki/Building-and-installing"]
    from repo_parse.ReadPrompt import URL_SUMMARY_SYS, URL_SUMMARY_USR
    asyncio.run(summarize_urls("", urls, URL_SUMMARY_SYS, URL_SUMMARY_USR))
    
