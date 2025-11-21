#!/usr/bin/env python3
"""
eucerin_full_crawler.py
Crawl Eucerin product pages, extract structured data, write JSONL per product.

Usage:
    python eucerin_full_crawler.py --output eucerin_products_full.jsonl
"""

import asyncio
import json
import re
import sys
import time
from argparse import ArgumentParser
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dateutil import tz
from playwright.async_api import async_playwright
from tqdm.asyncio import tqdm

# --- CONFIG (edit if needed) ---
SEED_PAGES = [
    "https://int.eucerin.com/products",
    "https://int.eucerin.com/",
]
BASE_DOMAIN = "int.eucerin.com"
USER_AGENT = "EucerinFullCrawler/1.0"
CONCURRENT_TASKS = 6
RATE_LIMIT_SECONDS = 0.6   # delay between requests per worker
MAX_PAGES = None           # set to an int to limit pages for testing
REQUEST_TIMEOUT = 30
OUTPUT_ENCODING = "utf-8"
# --------------------------------

import urllib.robotparser
ROBOTS_URL = f"https://{BASE_DOMAIN}/robots.txt"

def is_allowed_by_robots(url):
    try:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(ROBOTS_URL)
        rp.read()
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        # If robots cannot be read, be conservative and allow seed domain only
        parsed = urlparse(url)
        return parsed.netloc.endswith(BASE_DOMAIN)

def now_iso_date():
    # use local timezone awareness
    return time.strftime("%Y-%m-%d")

def normalize_url(base, href):
    if not href:
        return None
    href = href.split("#")[0]
    href = href.split("?")[0]
    return urljoin(base, href)

def find_sitemap_urls():
    # quick attempt: fetch robots.txt and parse Sitemap entries
    s_urls = []
    try:
        r = requests.get(ROBOTS_URL, headers={"User-Agent": USER_AGENT}, timeout=10)
        if r.status_code == 200:
            for line in r.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    s_urls.append(line.split(":",1)[1].strip())
    except Exception:
        pass
    return s_urls

def extract_product_links_from_html(html, base):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.select("a"):
        href = a.get("href")
        if not href: continue
        url = normalize_url(base, href)
        if not url: continue
        # heuristic: product pages often have '/products/' in the path
        if "/products/" in url:
            if BASE_DOMAIN in urlparse(url).netloc:
                links.add(url)
    # also attempt to parse JSON-LD for URLs
    for script in soup.select("script[type='application/ld+json']"):
        try:
            j = json.loads(script.string)
            # if it's a list
            if isinstance(j, list):
                for item in j:
                    if isinstance(item, dict) and item.get("@type","").lower() in ("product","productcollection"):
                        url = item.get("url")
                        if url:
                            url = normalize_url(base, url)
                            if "/products/" in url:
                                links.add(url)
            else:
                if isinstance(j, dict) and j.get("@type","").lower() in ("product","productcollection"):
                    url = j.get("url")
                    if url:
                        url = normalize_url(base, url)
                        if "/products/" in url:
                            links.add(url)
        except Exception:
            pass
    return links

async def fetch_page_content(playwright, url):
    if not is_allowed_by_robots(url):
        # skip disallowed URL
        return None
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context(user_agent=USER_AGENT, locale="en-US")
    page = await context.new_page()
    try:
        await page.goto(url, wait_until="networkidle", timeout=REQUEST_TIMEOUT * 1000)
        # small additional wait in case of lazy-loading images or sections
        await asyncio.sleep(0.15)
        content = await page.content()
        return content
    except Exception as e:
        print(f"[WARN] failed to render {url}: {e}", file=sys.stderr)
        return None
    finally:
        try:
            await context.close()
            await browser.close()
        except Exception:
            pass

def extract_structured_fields(html, url):
    soup = BeautifulSoup(html, "html.parser")

    # Title / product name heuristics
    title = None
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)

    # short description - look for hero intro / description blocks
    short_desc = ""
    selectors = [
        ".product-hero__description", ".product-intro", ".product-description",
        ".product-hero__subtitle", ".product-info__description"
    ]
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            short_desc = el.get_text(" ", strip=True)
            break
    if not short_desc:
        # fallback to first paragraph under h1
        if h1:
            p = h1.find_next("p")
            if p:
                short_desc = p.get_text(" ", strip=True)

    # full textual content (for embedding) - grab main article / product detail blocks
    main_text = []
    for css in ["main", ".product-main", ".product-detail", ".product-page"]:
        el = soup.select_one(css)
        if el:
            main_text.append(el.get_text(" ", strip=True))
    if not main_text:
        main_text.append(soup.get_text(" ", strip=True))

    full_text = "\n\n".join(main_text)
    # Try to extract ingredients: look for "Ingredients" heading or similar
    key_ingredients = []
    for heading in soup.find_all(re.compile("^h[1-6]$")):
        txt = heading.get_text(" ", strip=True).lower()
        if "ingredient" in txt:
            # collect following sibling text nodes up to a reasonable limit
            parts = []
            for sib in heading.find_next_siblings(limit=6):
                txt_sib = sib.get_text(" ", strip=True)
                if txt_sib:
                    parts.append(txt_sib)
            cand = " ".join(parts)
            # split by commas and semicolons
            for part in re.split(r",|;", cand):
                p = part.strip()
                if p and len(p) < 120:
                    key_ingredients.append(p)
            break

    # fallback: look for "Key ingredients" labels
    if not key_ingredients:
        labels = soup.find_all(text=re.compile(r"Key ingredient|Key ingredients|Active ingredient", re.I))
        for lab in labels[:3]:
            parent = lab.parent
            txt = parent.get_text(" ", strip=True)
            # attempt to extract after ":" or a known label
            if ":" in txt:
                tail = txt.split(":", 1)[1]
            else:
                tail = txt
            for part in re.split(r",|;", tail):
                p = part.strip()
                if p:
                    key_ingredients.append(p)

    # problems / skin concerns: many pages list "suitable for", "helps", "targets" etc.
    problems = set()
    for keyword in ["dry", "sensitive", "acne", "eczema", "psoriasis", "aging", "pigment", "hyperpigmentation", "redness", "itch", "irritation", "oily", "combination", "dehydrated"]:
        if re.search(r"\b" + re.escape(keyword) + r"\b", full_text, re.I):
            problems.add(keyword)

    # category detection by breadcrumbs or meta
    category = ""
    bc = soup.select_one(".breadcrumb, .breadcrumbs")
    if bc:
        category = bc.get_text(" ", strip=True)

    # last_seen timestamp
    last_seen = now_iso_date()

    return {
        "product_name": title or "",
        "short_description": short_desc or "",
        "key_ingredients": list(dict.fromkeys([k.strip() for k in key_ingredients if k.strip()])),
        "problems_solved": list(sorted(problems)),
        "category": category,
        "brand": "Eucerin",
        "text": full_text[:15000],   # cap per-document text to 15k chars to avoid massive embeddings
        "source": url,
        "last_seen": last_seen
    }

def make_id_from_url(url):
    p = urlparse(url)
    path = p.path.strip("/").replace("/", "_")
    id_safe = re.sub(r"[^a-zA-Z0-9_\-]", "", path)
    if not id_safe:
        id_safe = re.sub(r"[^a-zA-Z0-9]", "", p.netloc)
    return f"eucerin_{id_safe}"

async def worker(name, queue, seen_urls, output_file, playwright):
    while True:
        try:
            url = await queue.get()
        except asyncio.CancelledError:
            return
        if url is None:
            queue.task_done()
            return
        if url in seen_urls:
            queue.task_done()
            continue

        # rate-limit per worker
        await asyncio.sleep(RATE_LIMIT_SECONDS)

        content = await fetch_page_content(playwright, url)
        if content:
            data = extract_structured_fields(content, url)
            data["id"] = make_id_from_url(url)
            # write line
            with open("full.txt", "a", encoding=OUTPUT_ENCODING) as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            seen_urls.add(url)
        queue.task_done()

async def main(output_file, resume=False, max_pages=None):
    # resume: read existing output file and build seen set
    seen_urls = set()
    if resume:
        try:
            with open(output_file, "r", encoding=OUTPUT_ENCODING) as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        seen_urls.add(obj.get("source") or obj.get("url") or "")
                    except Exception:
                        continue
            print(f"[INFO] Resuming; {len(seen_urls)} URLs already processed")
        except FileNotFoundError:
            pass

    # seed discovery: sitemap + seed pages
    to_visit = set(SEED_PAGES)
    sitemap_urls = find_sitemap_urls()
    for s in sitemap_urls:
        try:
            r = requests.get(s, headers={"User-Agent": USER_AGENT}, timeout=10)
            if r.status_code == 200:
                # very basic sitemap parsing: find all hrefs
                for m in re.findall(r"<loc>(.*?)</loc>", r.text):
                    if "/products/" in m and BASE_DOMAIN in urlparse(m).netloc:
                        to_visit.add(m)
        except Exception:
            pass

    # start playwright and discover links from seed pages (non-rendered quick fetch first)
    discovered = set()
    for seed in list(to_visit):
        try:
            r = requests.get(seed, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                links = extract_product_links_from_html(r.text, seed)
                discovered.update(links)
        except Exception:
            pass

    all_start_urls = list(set(to_visit) | discovered)
    print(f"[INFO] Starting crawl with {len(all_start_urls)} seed pages (discovered {len(discovered)} product candidate links)")

    # Create queue of product URLs to scrape
    product_urls = set()
    # If discovered links look
