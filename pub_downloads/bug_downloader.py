import argparse
import os
from Bio import Entrez

def get_bug_information(bug, email, repository):
    """Fetch definition or functionality information about a specified bug from PubMed."""
    Entrez.email = email  # Always provide your email address
    search_term = f"{bug}[Title] AND review[Publication Type]"

    # Search for articles in PubMed
    search_handle = Entrez.esearch(db="pubmed", term=search_term, retmax=5)
    search_results = Entrez.read(search_handle)
    search_handle.close()

    # Get IDs of articles
    article_ids = search_results['IdList']
    print(f"PubMed IDs of articles on {bug}:", article_ids)

    # Fetch details of articles
    fetch_handle = Entrez.efetch(db="pubmed", id=",".join(article_ids), rettype="abstract", retmode="text")
    data = fetch_handle.read()
    fetch_handle.close()

    # Ensure repository exists, create if not
    if not os.path.exists(repository):
        os.makedirs(repository)

    # Create a filename from the bug name, replacing spaces with underscores
    filename = os.path.join(repository, f"{bug.replace(' ', '_')}_info.txt")

    # Save the data to a file
    with open(filename, "w") as file:
        file.write(data)

    print(f"Information about {bug} has been saved to {filename}.")

def main():
    parser = argparse.ArgumentParser(description="Fetch information about a specified bug from PubMed")
    parser.add_argument("--bug_name", type=str, default="Escherichia coli", help="Name of the bug to search")
    parser.add_argument("--email", type=str, default="first.name@university.edu", help="Email address to use with Entrez")
    parser.add_argument("--repository", type=str, default="bug_info", help="Directory to save the fetched information")
    args = parser.parse_args()

    # Call the function with command line arguments
    get_bug_information(args.bug_name, args.email, args.repository)

if __name__ == "__main__":
    main()
