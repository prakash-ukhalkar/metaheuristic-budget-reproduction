"""Resolve and verify every reference DOI against Crossref.

Run this on a machine with internet access:

    pip install requests
    python3 verify_references.py

For each entry it queries the Crossref REST API by title (and first author),
prints the best match with its DOI, similarity score and open-access status,
and flags anything where the returned title does not closely match the one on
record. Entries already carrying a verified DOI are re-checked rather than
trusted.

Nothing here writes to the manuscript. Copy the confirmed DOIs across by hand
after reading the output - an automated rewrite is exactly how a wrong DOI
gets into a published reference list.
"""
import difflib
import json
import sys
import time
import urllib.parse
import urllib.request

MAILTO = "your.email@institution.edu"   # Crossref asks for a contact address
API = "https://api.crossref.org/works"

# (key, title, first author surname, year, DOI if already verified else None)
REFS = [
 ("1", "Metaheuristics-the metaphor exposed", "Sorensen", 2015, "10.1111/itor.12001"),
 ("2", "Exposing the grey wolf, moth-flame, whale, firefly, bat, and antlion algorithms: six misleading optimization techniques inspired by bestial metaphors", "Camacho-Villalon", 2023, "10.1111/itor.13176"),
 ("3", "A critical problem in benchmarking and analysis of evolutionary computation methods", "Kudela", 2022, None),
 ("4", "Some metaheuristics should be simplified", "Piotrowski", 2018, None),
 ("5", "Parameter tuning for configuring and analyzing evolutionary algorithms", "Eiben", 2011, None),
 ("6", "A rigorous analysis of the harmony search algorithm", "Weyland", 2010, "10.4018/jamc.2010040104"),
 ("7", "Metaphor-based metaheuristics, a call for action: the elephant in the room", "Aranha", 2022, "10.1007/s11721-021-00202-9"),
 ("8", "Statistical comparisons of classifiers over multiple data sets", "Demsar", 2006, None),
 ("9", "An extension on statistical comparisons of classifiers over multiple data sets for all pairwise comparisons", "Garcia", 2008, None),
 ("10", "A practical guide for using statistical tests to assess randomized algorithms in software engineering", "Arcuri", 2014, None),
 ("11", "COCO: a platform for comparing continuous optimizers in a black-box setting", "Hansen", 2021, None),
 ("12", "Benchmarking in optimization: best practice and open issues", "Bartz-Beielstein", 2020, None),
 ("14", "Nonlinear integer and discrete programming in mechanical design optimization", "Sandgren", 1990, None),
 ("15", "An augmented Lagrange multiplier based method for mixed integer discrete continuous optimization and its applications to mechanical design", "Kannan", 1994, None),
 ("16", "Use of a self-adaptive penalty approach for engineering optimization problems", "Coello", 2000, None),
 ("17", "A test-suite of non-convex constrained optimization problems from the real-world and some baseline results", "Kumar", 2020, "10.1016/j.swevo.2020.100693"),
 ("18", "An efficient constraint handling method for genetic algorithms", "Deb", 2000, "10.1016/S0045-7825(99)00389-8"),
 ("19", "Constrained optimization by the epsilon constrained differential evolution with gradient-based mutation and feasible elites", "Takahama", 2006, None),
 ("20", "Grey wolf optimizer", "Mirjalili", 2014, "10.1016/j.advengsoft.2013.12.007"),
 ("21", "The whale optimization algorithm", "Mirjalili", 2016, "10.1016/j.advengsoft.2016.01.008"),
 ("22", "SCA: a sine cosine algorithm for solving optimization problems", "Mirjalili", 2016, "10.1016/j.knosys.2015.12.022"),
 ("23", "Salp swarm algorithm: a bio-inspired optimizer for engineering design problems", "Mirjalili", 2017, "10.1016/j.advengsoft.2017.07.002"),
 ("24", "Harris hawks optimization: algorithm and applications", "Heidari", 2019, "10.1016/j.future.2019.02.028"),
 ("25", "The arithmetic optimization algorithm", "Abualigah", 2021, "10.1016/j.cma.2020.113609"),
 ("26", "Differential evolution - a simple and efficient heuristic for global optimization over continuous spaces", "Storn", 1997, None),
 ("27", "The particle swarm - explosion, stability, and convergence in a multidimensional complex space", "Clerc", 2002, None),
 ("28", "Improving the search performance of SHADE using linear population size reduction", "Tanabe", 2014, None),
 ("29", "Completely derandomized self-adaptation in evolution strategies", "Hansen", 2001, None),
 ("30", "The irace package: iterated racing for automatic algorithm configuration", "Lopez-Ibanez", 2016, None),
 ("31", "A simple sequentially rejective multiple test procedure", "Holm", 1979, None),
 ("32", "A critique and improvement of the CL common language effect size statistics of McGraw and Wong", "Vargha", 2000, None),
]

# References 13 (Arora, book) and 8/9 (JMLR) may have no Crossref DOI.
# JMLR is fully open access at jmlr.org; cite the URL instead.


def query(title, author):
    params = {"query.bibliographic": title, "query.author": author,
              "rows": 3, "mailto": MAILTO}
    url = f"{API}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)["message"]["items"]


def main():
    print(f"{'ref':<5}{'status':<10}{'sim':>6}  DOI / note")
    print("-" * 100)
    for key, title, author, year, known in REFS:
        try:
            items = query(title, author)
        except Exception as e:
            print(f"[{key}]  ERROR     {'':>6}  {e}")
            continue
        if not items:
            print(f"[{key}]  NOT FOUND {'':>6}  search manually: {title[:60]}")
            continue
        best = items[0]
        got_title = (best.get("title") or [""])[0]
        sim = difflib.SequenceMatcher(None, title.lower(), got_title.lower()).ratio()
        doi = best.get("DOI", "")
        oa = "open" if best.get("license") else "check access"
        if known and doi.lower() == known.lower():
            status = "CONFIRMED"
        elif known:
            status = "MISMATCH!"
        elif sim > 0.85:
            status = "RESOLVED"
        else:
            status = "LOW MATCH"
        print(f"[{key}]  {status:<10}{sim:>6.2f}  {doi}   [{oa}]")
        if status in ("MISMATCH!", "LOW MATCH"):
            print(f"{'':<21}expected: {title[:70]}")
            print(f"{'':<21}returned: {got_title[:70]}")
        time.sleep(0.6)   # be polite to the Crossref API


if __name__ == "__main__":
    if MAILTO.startswith("your.email"):
        print("Set MAILTO to your address first (Crossref requests it).", file=sys.stderr)
    main()
