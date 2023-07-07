import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files.
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus.
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Overall time complexity for transition_model: O(N).

    # Create a probability distribution dictionary.
    distribution = {}

    # Get the total number of pages in the corpus.
    total_pages = len(corpus)

    # If the current page has no outgoing links, return a probability.
    # distribution that chooses randomly from all pages in the corpus.
    if len(corpus[page]) == 0:
        return {p: 1 / total_pages for p in corpus}

    # Otherwise, create the probability distribution as described above.
    for p in corpus:
        if p in corpus[page]:
            distribution[p] = damping_factor / len(corpus[page])
        else:
            distribution[p] = (1 - damping_factor) / total_pages

    return distribution


def sample_pagerank(corpus, damping, num_samples):
    """
    Samples num_samples transitions from the given corpus and computes
    PageRank over the resulting sample.

    Returns a dictionary where the keys are page names and the values
    are their corresponding PageRank values.
    """
    # Overall time complexity for sample_pagerank: O(n*m).

    # 1. Time complexity: O(n*m).
    # Collect counts of transitions from each page in the corpus.
    counts = {}
    for page in corpus:
        for link in corpus[page]:
            if page not in counts:
                counts[page] = {link: 1}
            else:
                if link not in counts[page]:
                    counts[page][link] = 1
                else:
                    counts[page][link] += 1

    # 2. Time complexity: O(n*m).
    # Convert counts to probabilities.
    for page in counts:
        total = sum(counts[page].values())
        for link in counts[page]:
            counts[page][link] /= total

    # 3. Time complexity: O(num_samples)
    # Sample transitions and compute PageRank.
    pagerank = {}
    current_page = random.choice(list(corpus.keys()))
    for i in range(num_samples):
        pagerank[current_page] = pagerank.get(current_page, 0) + 1
        if random.random() < damping:
            next_page = random.choice(list(corpus.keys()))
        else:
            if current_page in counts:
                next_page = random.choices(list(counts[current_page].keys()), weights=list(
                    counts[current_page].values()))[0]
            else:
                next_page = random.choice(list(corpus.keys()))
        current_page = next_page

    # 4. Time complexity: O(n).
    # Normalize the PageRank values.
    total = sum(pagerank.values())
    for page in pagerank:
        pagerank[page] /= total

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Overall time complexity for iterate_pagerank: O(N^3).

    N = len(corpus)
    d = damping_factor
    threshold = 0.001 / N

    # Initialize all page ranks to 1/N
    page_ranks = dict.fromkeys(corpus.keys(), 1 / N)

    while True:
        new_page_ranks = {}

        # Calculate the PageRank of each page
        for page in corpus:
            pr = ((1 - d) / N) + d * sum(page_ranks[link] / len(corpus[link])
                                         for link in corpus if page in corpus[link])
            new_page_ranks[page] = pr

        # Check for convergence
        diff = sum(abs(new_page_ranks[page] - page_ranks[page])
                   for page in corpus)
        if diff < threshold:
            break

        page_ranks = new_page_ranks

    # Normalize the Page Ranks so that they sum to 1
    normalization_factor = sum(page_ranks[page] for page in corpus)
    page_ranks = {page: page_ranks[page] /
                  normalization_factor for page in corpus}

    return page_ranks


if __name__ == "__main__":
    main()
