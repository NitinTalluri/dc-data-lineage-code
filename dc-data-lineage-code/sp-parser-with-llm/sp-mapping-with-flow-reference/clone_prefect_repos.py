#!/usr/bin/env python3
"""
Script to clone only Prefect flow repositories from AWS CodeCommit.
Uses credentials.txt with Username/Password format for AWS authentication.
uv run pyhton cloe_prefect_repos.py
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Set
import re
from urllib.parse import quote
import boto3

class PrefectRepoCloner:
    def __init__(self, credentials_file: str = "credentials.txt"):
        self.credentials_file = credentials_file
        self.credentials = {}
        self.prefect_patterns = {
            'files': [
                'prefect.yaml',
                'prefect.dev.yaml', 
                'prefect.prod.yaml',
                'prefect.staging.yaml'
            ],
            'code_patterns': [
                r'from prefect import',
                r'import prefect',
                r'@flow\b',
                r'@task\b',
                r'deployment_tags',
                r'flow_service',
                r'FlowService',
                r'prefect\.flow',
                r'prefect\.task',
                r'common_prefect_next',
                r'entrypoint.*flow'
            ],
            'repo_name_patterns': [
                r'.*-flows$',
                r'.*-flow$',
                r'_flows$',
                r'_flow$',
                r'flow-.*',
                r'flow_.*',
                r'.*flows.*',
                r'prefect-.*',
                r'.*-prefect.*'
            ]
        }
        
    def _load_credentials(self) -> Dict[str, str]:
        """Load AWS CodeCommit credentials from file using same method as quick_table.py."""
        try:
            credentials = {}
            with open(self.credentials_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip().strip('"')
                        value = value.strip().strip('"')
                        credentials[key.lower()] = value
            
            print(f"Loaded credentials from {self.credentials_file}")
            return credentials
        except FileNotFoundError:
            print(f"Error: Credentials file {self.credentials_file} not found!")
            return {}
        except Exception as e:
            print(f"Error loading credentials: {e}")
            return {}
    
    def setup_aws_credentials(self):
        """Setup AWS credentials from credentials.txt file."""
        self.credentials = self._load_credentials()
        
        if not self.credentials.get('username') or not self.credentials.get('password'):
            print("Error: Username or Password not found in credentials file")
            sys.exit(1)
        
        print("Git credentials loaded successfully")
    
    def get_all_repositories_via_boto3(self) -> List[Dict]:
        """Get repositories using boto3 (same approach as quick_table.py)."""
        try:
            print("Getting repository list via boto3...")
            
            # Initialize boto3 client for CodeCommit
            session = boto3.Session(region_name='us-east-1')
            codecommit = session.client('codecommit')
            
            # List repositories
            response = codecommit.list_repositories()
            repositories = response.get('repositories', [])
            
            print(f"Found {len(repositories)} total repositories")
            return repositories
            
        except Exception as e:
            print(f"Error getting repositories via boto3: {e}")
            print("Falling back to AWS CLI...")
            return self.get_all_repositories_via_cli()
    
    def get_all_repositories_via_cli(self) -> List[Dict]:
        """Get repositories using AWS CLI (fallback method)."""
        try:
            print("Getting repository list via AWS CLI...")
            result = subprocess.run(['aws', 'codecommit', 'list-repositories'], 
                                  capture_output=True, text=True, check=True)
            import json
            data = json.loads(result.stdout)
            repositories = data.get('repositories', [])
            print(f"Found {len(repositories)} total repositories")
            return repositories
        except subprocess.CalledProcessError as e:
            print(f"Error running AWS CLI: {e}")
            print("Make sure AWS CLI is installed and configured")
            return []
        except Exception as e:
            print(f"Error parsing AWS CLI output: {e}")
            return []
    
    def filter_by_naming_convention(self, repositories: List[Dict]) -> Set[str]:
        """Filter repositories by naming patterns."""
        matching_repos = set()
        
        for repo in repositories:
            repo_name = repo['repositoryName']
            for pattern in self.prefect_patterns['repo_name_patterns']:
                if re.match(pattern, repo_name, re.IGNORECASE):
                    matching_repos.add(repo_name)
                    print(f"Found by naming pattern: {repo_name}")
                    break
        
        return matching_repos
    
    def get_clone_url_with_auth(self, repo_name: str, region: str = "us-east-1") -> str:
        """Get the clone URL with properly encoded authentication (same as quick_table.py)."""
        username = self.credentials.get('username', '')
        password = self.credentials.get('password', '')
        
        # Handle special characters in username and password (same as quick_table.py)
        encoded_username = quote(username, safe='')
        encoded_password = quote(password, safe='')
        
        clone_url_https = f"https://git-codecommit.{region}.amazonaws.com/v1/repos/{repo_name}"
        auth_url = clone_url_https.replace('https://', f'https://{encoded_username}:{encoded_password}@')
        
        return auth_url
    
    
    def shallow_clone_and_check(self, repo_name: str, region: str = "us-east-1") -> bool:
        """Perform shallow clone and check for Prefect patterns."""
        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp(prefix=f"prefect_check_{repo_name}_")
            clone_url = self.get_clone_url_with_auth(repo_name, region)
            
            print(f"Checking repository: {repo_name}")
            print(f"  Cloning (timeout: 60s)...")
            
            # Use git command with proper error handling and shorter timeout
            cmd = ['git', 'clone', '--depth', '1', clone_url, temp_dir]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)  # Reduced from 300
            
            if result.returncode != 0:
                print(f"  Could not clone {repo_name}: {result.stderr.strip()}")
                return False
            
            print(f"  Clone successful, checking for Prefect patterns...")
            
            # Check for Prefect patterns
            if self._check_prefect_files(temp_dir):
                return True
            
            if self._check_prefect_code_patterns(temp_dir):
                return True
            
            print(f"  No Prefect patterns found in {repo_name}")
            return False
            
        except subprocess.TimeoutExpired:
            print(f"  Timeout (60s) checking {repo_name} - skipping")
            return False
        except Exception as e:
            print(f"  Error checking {repo_name}: {e}")
            return False
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _check_prefect_files(self, repo_path: str) -> bool:
        """Check for Prefect configuration files."""
        for file_pattern in self.prefect_patterns['files']:
            if list(Path(repo_path).rglob(file_pattern)):
                print(f"  Found Prefect config file: {file_pattern}")
                return True
        return False
    
    def _check_prefect_code_patterns(self, repo_path: str) -> bool:
        """Check for Prefect code patterns in Python files."""
        python_files = list(Path(repo_path).rglob("*.py"))
        
        # Limit to first 50 Python files to avoid excessive checking
        for py_file in python_files[:50]:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                for pattern in self.prefect_patterns['code_patterns']:
                    if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                        print(f"  Found Prefect pattern '{pattern}' in {py_file.name}")
                        return True
            except Exception:
                continue
        
        return False
    
    def clone_repository(self, repo_name: str, target_dir: str = "prefect_repos", region: str = "us-east-1"):
        """Clone a repository to the target directory."""
        try:
            os.makedirs(target_dir, exist_ok=True)
            repo_path = os.path.join(target_dir, repo_name)
            
            if os.path.exists(repo_path):
                print(f"Repository {repo_name} already exists, pulling latest changes...")
                try:
                    # Try to pull latest changes (same as quick_table.py)
                    result = subprocess.run(
                        ['git', 'pull'],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    if result.returncode == 0:
                        print(f"Successfully updated {repo_name}")
                    else:
                        print(f"Could not update {repo_name}, using existing version")
                    return True
                except Exception as e:
                    print(f"Could not update {repo_name}: {e}, using existing version")
                    return True
            
            print(f"Cloning {repo_name}...")
            clone_url = self.get_clone_url_with_auth(repo_name, region)
            
            # Use same approach as quick_table.py
            cmd = ['git', 'clone', clone_url, repo_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"Successfully cloned {repo_name}")
                return True
            else:
                print(f"Error cloning {repo_name}: {result.stderr.strip()}")
                return False
            
        except subprocess.TimeoutExpired:
            print(f"Timeout cloning {repo_name}")
            return False
        except Exception as e:
            print(f"Error cloning {repo_name}: {e}")
            return False
    
    def run(self, target_dir: str = "prefect_repos", skip_naming_check: bool = False, 
            region: str = "us-east-1", repo_list: List[str] = None):
        """Main execution method."""
        print("Starting Prefect repository discovery and cloning...")
        
        self.setup_aws_credentials()
        
        if repo_list:
            all_repos = [{'repositoryName': name} for name in repo_list]
            print(f"Using provided repository list: {len(all_repos)} repositories")
        else:
            # Try boto3 first, then fall back to CLI (same as quick_table.py)
            all_repos = self.get_all_repositories_via_boto3()
            if not all_repos:
                print("No repositories found. Please check your AWS configuration.")
                return
        
        prefect_repos = set()
        
        # Step 1: Filter by naming convention
        if not skip_naming_check:
            print("\nStep 1: Filtering by naming conventions...")
            prefect_repos.update(self.filter_by_naming_convention(all_repos))
        
        # Step 2: Check remaining repositories by content
        print("\nStep 2: Checking repositories for Prefect patterns...")
        
        remaining_repos = [r for r in all_repos if r['repositoryName'] not in prefect_repos]
        
        for i, repo in enumerate(remaining_repos, 1):
            repo_name = repo['repositoryName']
            print(f"[{i}/{len(remaining_repos)}] Checking {repo_name}")
            
            if self.shallow_clone_and_check(repo_name, region):
                prefect_repos.add(repo_name)
                print(f"  -> {repo_name} contains Prefect flows")
        
        # Step 3: Clone identified Prefect repositories
        print(f"\nStep 3: Cloning {len(prefect_repos)} Prefect repositories...")
        
        successful_clones = 0
        for i, repo_name in enumerate(sorted(prefect_repos), 1):
            print(f"[{i}/{len(prefect_repos)}] ", end="")
            if self.clone_repository(repo_name, target_dir, region):
                successful_clones += 1
        
        # Summary
        print(f"\nSummary:")
        print(f"  Total repositories: {len(all_repos)}")
        print(f"  Prefect repositories found: {len(prefect_repos)}")
        print(f"  Successfully cloned: {successful_clones}")
        print(f"  Cloned to directory: {os.path.abspath(target_dir)}")
        
        if prefect_repos:
            print(f"\nPrefect repositories:")
            for repo in sorted(prefect_repos):
                print(f"  - {repo}")
        
        # Save list of Prefect repositories for future reference
        prefect_list_file = os.path.join(target_dir, "prefect_repositories.txt")
        try:
            with open(prefect_list_file, 'w') as f:
                for repo in sorted(prefect_repos):
                    f.write(f"{repo}\n")
            print(f"\nPrefect repository list saved to: {prefect_list_file}")
        except Exception as e:
            print(f"Could not save repository list: {e}")


def main():
    """Main function with command line argument support."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clone Prefect flow repositories from AWS CodeCommit")
    parser.add_argument("--credentials", default="credentials.txt", 
                       help="Path to credentials file (default: credentials.txt)")
    parser.add_argument("--target-dir", default="prefect_repos",
                       help="Target directory for cloned repositories (default: prefect_repos)")
    parser.add_argument("--region", default="us-east-1",
                       help="AWS region (default: us-east-1)")
    parser.add_argument("--skip-naming-check", action="store_true",
                       help="Skip naming convention filtering (check all repos)")
    parser.add_argument("--repo-list", nargs="+",
                       help="Specific repository names to check (space-separated)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Only identify repositories, don't clone them")
    
    args = parser.parse_args()
    
    cloner = PrefectRepoCloner(credentials_file=args.credentials)
    
    if args.dry_run:
        print("DRY RUN MODE - Only identifying repositories...")
        original_clone = cloner.clone_repository
        def dry_run_clone(repo_name, target_dir, region):
            print(f"[DRY RUN] Would clone: {repo_name}")
            return True
        cloner.clone_repository = dry_run_clone
    
    cloner.run(
                target_dir=args.target_dir, 
        skip_naming_check=args.skip_naming_check,
        region=args.region,
        repo_list=args.repo_list
    )


if __name__ == "__main__":
    main()