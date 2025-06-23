#!/usr/bin/env python3
"""
Robust PMC Article Downloader with intelligent retry logic
Handles HTTP 429 errors and implements adaptive rate limiting

Install: pip install requests tqdm
"""

import argparse
import requests
import os
import time
import logging
from glob import glob
import re
import xml.etree.ElementTree as ET
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore, Lock
import random
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on server responses"""
    def __init__(self, initial_rate=0.34, min_rate=0.1, max_rate=2.0):
        self.rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.lock = Lock()
        self.last_request_time = 0
        self.consecutive_429s = 0
        self.consecutive_successes = 0
        
    def wait(self):
        """Wait appropriate time before next request"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate:
                sleep_time = self.rate - time_since_last
                time.sleep(sleep_time)
            self.last_request_time = time.time()
    
    def report_success(self):
        """Report successful request to potentially speed up"""
        with self.lock:
            self.consecutive_429s = 0
            self.consecutive_successes += 1
            
            # Speed up after many successes
            if self.consecutive_successes > 50 and self.rate > self.min_rate:
                self.rate = max(self.rate * 0.9, self.min_rate)
                logger.debug(f"Speeding up: new rate = {self.rate:.3f}s")
    
    def report_429(self):
        """Report 429 error to slow down"""
        with self.lock:
            self.consecutive_successes = 0
            self.consecutive_429s += 1
            
            # Exponential backoff for rate
            self.rate = min(self.rate * (1.5 ** self.consecutive_429s), self.max_rate)
            logger.info(f"Got 429 error, slowing down: new rate = {self.rate:.3f}s")
            
            # Return extra wait time for this specific request
            return self.rate * (2 ** self.consecutive_429s)


class RobustPMCDownloader:
    def __init__(self, api_key="", max_workers=5, initial_rate=0.34):
        """
        Initialize downloader with advanced retry capabilities.
        
        Args:
            api_key: NCBI API key
            max_workers: Number of concurrent download threads
            initial_rate: Initial rate limit in seconds
        """
        self.api_key = api_key
        self.max_workers = max_workers
        self.rate_limiter = AdaptiveRateLimiter(
            initial_rate=0.1 if api_key else 0.34,
            min_rate=0.1 if api_key else 0.34
        )
        self.session = self._create_session()
        self.failed_downloads = {}  # Track failures with retry counts
        
    def _create_session(self):
        """Create requests session with retry adapter"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'PMCDownloader/1.0 (Python requests)'
        })
        return session
    
    def search_articles(self, query, field, max_results=100000):
        """Search for articles with robust error handling"""
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        ids = []
        search_term = f"({query}[{field}])"
        
        # Get total count with retries
        for attempt in range(3):
            try:
                self.rate_limiter.wait()
                params = {
                    'db': 'pmc',
                    'term': search_term,
                    'retmax': 0,
                    'api_key': self.api_key
                }
                
                response = self.session.get(base_url, params=params, timeout=30)
                
                if response.status_code == 429:
                    wait_time = self.rate_limiter.report_429()
                    logger.warning(f"Rate limited on search, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                root = ET.fromstring(response.content)
                total_count = int(root.find('.//Count').text)
                logger.info(f"Total articles found: {total_count}")
                self.rate_limiter.report_success()
                break
                
            except Exception as e:
                if attempt == 2:
                    logger.error(f"Failed to get article count: {e}")
                    return []
                time.sleep(2 ** attempt)
        
        # Fetch IDs in batches
        batch_size = 5000  # Smaller batches to avoid timeouts
        for start in tqdm(range(0, min(total_count, max_results), batch_size), 
                         desc="Fetching PMC IDs"):
            
            for attempt in range(3):
                try:
                    self.rate_limiter.wait()
                    
                    params = {
                        'db': 'pmc',
                        'term': search_term,
                        'retstart': start,
                        'retmax': min(batch_size, max_results - start),
                        'api_key': self.api_key
                    }
                    
                    response = self.session.get(base_url, params=params, timeout=45)
                    
                    if response.status_code == 429:
                        wait_time = self.rate_limiter.report_429()
                        logger.warning(f"Rate limited, waiting {wait_time:.1f}s")
                        time.sleep(wait_time)
                        continue
                        
                    response.raise_for_status()
                    root = ET.fromstring(response.content)
                    batch_ids = [id_.text for id_ in root.findall('.//Id')]
                    ids.extend(batch_ids)
                    self.rate_limiter.report_success()
                    
                    if len(batch_ids) < batch_size:
                        break
                    break
                    
                except Exception as e:
                    if attempt == 2:
                        logger.error(f"Error fetching batch at {start}: {e}")
                    else:
                        time.sleep(2 ** attempt)
                        
        return ids
    
    def download_article_with_retry(self, pmcid, output_directory, max_retries=5):
        """Download a single article with advanced retry logic"""
        file_path = os.path.join(output_directory, f"PMC{pmcid}.xml")
        
        # Check if already downloaded and valid
        if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:
            return True, "already_downloaded"
        
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            'db': 'pmc',
            'id': pmcid,
            'retmode': 'xml',
            'api_key': self.api_key
        }
        
        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait()
                
                # Add jitter to avoid thundering herd
                if attempt > 0:
                    jitter = random.uniform(0, 2 ** attempt)
                    time.sleep(jitter)
                
                response = self.session.get(url, params=params, timeout=60)
                
                if response.status_code == 200:
                    # Validate content before saving
                    content = response.content
                    if len(content) < 1000 or b'<error>' in content[:200]:
                        logger.warning(f"Invalid content for PMC{pmcid}, retrying...")
                        continue
                    
                    # Save file
                    with open(file_path, 'wb') as file:
                        file.write(content)
                    
                    self.rate_limiter.report_success()
                    return True, "success"
                    
                elif response.status_code == 429:
                    # Rate limited - wait longer
                    wait_time = self.rate_limiter.report_429()
                    logger.warning(f"Rate limited on PMC{pmcid}, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    continue
                    
                elif response.status_code == 404:
                    logger.warning(f"PMC{pmcid} not found (404)")
                    return False, "not_found"
                    
                else:
                    logger.warning(f"Failed PMC{pmcid}: HTTP {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout for PMC{pmcid} (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    
            except Exception as e:
                logger.error(f"Error downloading PMC{pmcid}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return False, "max_retries_exceeded"
    
    def download_articles_parallel(self, pmcids, output_directory):
        """Download articles in parallel with progress tracking"""
        os.makedirs(output_directory, exist_ok=True)
        
        # Statistics
        stats = {
            'success': 0,
            'already_downloaded': 0,
            'not_found': 0,
            'failed': 0,
            'max_retries': 0
        }
        
        failed_pmcids = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_pmcid = {
                executor.submit(self.download_article_with_retry, pmcid, output_directory): pmcid
                for pmcid in pmcids
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(pmcids), desc="Downloading articles") as pbar:
                for future in as_completed(future_to_pmcid):
                    pmcid = future_to_pmcid[future]
                    try:
                        success, reason = future.result(timeout=120)
                        
                        if success:
                            if reason == "already_downloaded":
                                stats['already_downloaded'] += 1
                            else:
                                stats['success'] += 1
                        else:
                            if reason == "not_found":
                                stats['not_found'] += 1
                            elif reason == "max_retries_exceeded":
                                stats['max_retries'] += 1
                                failed_pmcids.append(pmcid)
                            else:
                                stats['failed'] += 1
                                failed_pmcids.append(pmcid)
                                
                    except Exception as e:
                        logger.error(f"Error processing PMC{pmcid}: {e}")
                        stats['failed'] += 1
                        failed_pmcids.append(pmcid)
                    
                    pbar.update(1)
                    
                    # Update progress bar description with stats
                    pbar.set_postfix({
                        'OK': stats['success'],
                        'Skip': stats['already_downloaded'],
                        'Fail': stats['failed'] + stats['max_retries']
                    })
        
        return stats, failed_pmcids
    
    def save_failed_list(self, failed_pmcids, output_directory):
        """Save list of failed downloads for later retry"""
        if failed_pmcids:
            failed_file = os.path.join(output_directory, "failed_downloads.txt")
            with open(failed_file, 'w') as f:
                for pmcid in failed_pmcids:
                    f.write(f"{pmcid}\n")
            logger.info(f"Saved {len(failed_pmcids)} failed IDs to {failed_file}")
    
    def load_failed_list(self, output_directory):
        """Load list of previously failed downloads"""
        failed_file = os.path.join(output_directory, "failed_downloads.txt")
        if os.path.exists(failed_file):
            with open(failed_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        return []


def get_downloaded_ids(output_directory):
    """Get list of successfully downloaded PMC IDs"""
    downloaded_ids = set()
    
    if not os.path.exists(output_directory):
        return downloaded_ids
        
    for file_path in glob(os.path.join(output_directory, "PMC*.xml")):
        # Check file is valid
        if os.path.getsize(file_path) > 1000:
            match = re.search(r'PMC(\d+)\.xml', os.path.basename(file_path))
            if match:
                downloaded_ids.add(match.group(1))
                
    return downloaded_ids


def main():
    parser = argparse.ArgumentParser(
        description='Robust PMC Article Downloader with smart retry logic'
    )
    parser.add_argument('--query', type=str, required=True,
                        help='Search query')
    parser.add_argument('--field', type=str, default='Abstract',
                        choices=['Title', 'Abstract', 'Text Word'],
                        help='Field to search in')
    parser.add_argument('--output_directory', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--api_key', type=str, default='',
                        help='NCBI API key')
    parser.add_argument('--max_workers', type=int, default=5,
                        help='Number of download threads')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size for downloads')
    parser.add_argument('--max_results', type=int, default=100000,
                        help='Maximum search results')
    parser.add_argument('--retry_failed', action='store_true',
                        help='Only retry previously failed downloads')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_directory is None:
        args.output_directory = args.query.replace(' ', '_').lower()
    
    # Create output directory
    os.makedirs(args.output_directory, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.output_directory, "download.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting download session for query: '{args.query}'")
    
    # Initialize downloader with adaptive rate limiting
    downloader = RobustPMCDownloader(
        api_key=args.api_key,
        max_workers=args.max_workers
    )
    
    if args.retry_failed:
        # Only retry failed downloads
        ids_to_download = downloader.load_failed_list(args.output_directory)
        logger.info(f"Retrying {len(ids_to_download)} previously failed downloads")
    else:
        # Normal search and download
        logger.info(f"Searching for articles...")
        pmc_ids = downloader.search_articles(args.query, args.field, args.max_results)
        
        # Get already downloaded
        downloaded_ids = get_downloaded_ids(args.output_directory)
        
        # Calculate what to download
        pmc_ids_set = set(pmc_ids)
        ids_to_download = list(pmc_ids_set - downloaded_ids)
        
        logger.info(f"Total found: {len(pmc_ids)}")
        logger.info(f"Already downloaded: {len(downloaded_ids)}")
        logger.info(f"To download: {len(ids_to_download)}")
    
    if ids_to_download:
        all_failed = []
        
        # Download in batches
        for i in range(0, len(ids_to_download), args.batch_size):
            batch = ids_to_download[i:i + args.batch_size]
            logger.info(f"\nProcessing batch {i//args.batch_size + 1} ({len(batch)} articles)")
            
            stats, failed_pmcids = downloader.download_articles_parallel(batch, args.output_directory)
            all_failed.extend(failed_pmcids)
            
            # Log batch statistics
            logger.info(f"Batch stats: {stats}")
            
            # If getting many 429s, pause longer between batches
            if stats.get('max_retries', 0) > len(batch) * 0.1:
                logger.info("Many rate limit errors, pausing 30s between batches...")
                time.sleep(30)
            elif i + args.batch_size < len(ids_to_download):
                time.sleep(5)
        
        # Save final failed list
        downloader.save_failed_list(all_failed, args.output_directory)
        
        # Final summary
        final_downloaded = get_downloaded_ids(args.output_directory)
        logger.info(f"\nDownload session complete!")
        logger.info(f"Total successfully downloaded: {len(final_downloaded)}")
        logger.info(f"Failed downloads: {len(all_failed)}")
        
        if all_failed:
            logger.info(f"\nTo retry failed downloads, run:")
            logger.info(f"python {os.path.basename(__file__)} --query \"{args.query}\" "
                       f"--output_directory \"{args.output_directory}\" --retry_failed")
    else:
        logger.info("No articles to download!")


if __name__ == "__main__":
    main()
