import requests
import os

# Consider storing your API key securely (e.g., environment variable)
# You can request a key from the Semantic Scholar website for higher rate limits
# api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
api_key = "JlpfJSo5zL7lqQU8P7bEV9wsJzDYzNJk6XFqG2th" # Replace with your key if you have one

# Define the search query and parameters
search_query = "real-time speech driven facial animation"
search_fields = "title,abstract,tldr,authors,year,citationCount,venue,url"
search_year = "2025" # Focus on recent papers
search_limit = 50

# Construct the API request URL and parameters
base_url = "https://api.semanticscholar.org/graph/v1"
endpoint = "/paper/search"

params = {
    "query": search_query,
    "fields": search_fields,
    "year": search_year,
    "limit": search_limit,
    # Add offset for pagination if needed: 'offset': 0, 100, 200...
    # Add sort if needed and supported: 'sort': 'citationCount:desc'
}

headers = {}
if api_key:
    # API key must be in the header as 'x-api-key' (case-sensitive)
    headers['x-api-key'] = api_key

try:
    response = requests.get(base_url + endpoint, params=params, headers=headers)
    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

    results = response.json()

    print(f"Found {results.get('total', 0)} papers.")
    print(f"Showing offset {results.get('offset', 0)} to {results.get('offset', 0) + len(results.get('data', []))}")

    if 'data' in results:
        for paper in results['data']:
            print("-" * 20)
            print(f"Title: {paper.get('title')}")
            print(f"Year: {paper.get('year')}")
            print(f"Citations: {paper.get('citationCount')}")
            print(f"Venue: {paper.get('venue')}")
            print(f"URL: {paper.get('url')}")
            # Print authors if available
            authors = ", ".join([author.get('name', 'N/A') for author in paper.get('authors', [])])
            print(f"Authors: {authors}")
            # Print TLDR if available
            tldr = paper.get('tldr')
            if tldr:
              print(f"TLDR: {tldr.get('text')}")
            # print(f"Abstract: {paper.get('abstract')}") # Can be long

    # Handle pagination if total > limit
    if results.get('total', 0) > search_limit and 'next' in results:
        print("\nMore results available. Use offset parameter to retrieve.")
        print(f"Next offset would be: {results['next']}")


except requests.exceptions.RequestException as e:
    print(f"Error during API request: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response status code: {e.response.status_code}")
        print(f"Response text: {e.response.text}")