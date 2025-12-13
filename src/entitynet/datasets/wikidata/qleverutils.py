import sys
import requests
import time
from timeit import default_timer

from packg.misc import format_exception


QLEVER_URL = "https://qlever.dev/api/wikidata"


def query_qlever(query: str) -> list[list[str]]:
    response = requests.post(
        QLEVER_URL,
        headers={
            "Content-type": "application/sparql-query",
            "Accept": "text/tab-separated-values",
        },
        data=query,
    )
    assert response.status_code == 200, f"query failed: {response.text}"
    return [line.decode().split("\t") for line in response.iter_lines()][1:]


def query_qlever_robust(
    query: str,
    n_retries: int | None = 5,
    retry_delay_seconds: float = 5.0,
    timeout_seconds: float | None = None,
) -> list[list[str]]:
    """Retry query_qlever until it succeeds or configured limits are reached."""

    start = default_timer()
    attempt = 0
    last_exc = None

    while True:
        if timeout_seconds is not None and default_timer() > start + timeout_seconds:
            break
        if n_retries is not None and attempt >= n_retries:
            break

        attempt += 1
        try:
            return query_qlever(query)
        except (AssertionError, requests.RequestException) as e:
            print(f"qlever request failed, {attempt=} err: {format_exception(e)}", file=sys.stderr)
            last_exc = e
        sleep_time = retry_delay_seconds
        time.sleep(sleep_time)

    elapsed = default_timer() - start
    raise RuntimeError(
        f"query_qlever failed after {attempt} attempts and {elapsed:.1f}s"
    ) from last_exc


def query_living_entities(entity_ids: list[str]) -> list[list[str]]:
    """Query QLever for living entities by their Wikidata IDs e.g "wd:Q742292"."""
    query_body = LIVING_ENTITIES_QUERY.replace("__ENTITIES_PLACEHOLDER__", "\n".join(entity_ids))
    results = query_qlever_robust(query_body)
    return results


def query_living_entities_batched(entity_ids: list[str], batch_size: int = 500) -> list[list[str]]:
    """Query QLever for living entities by their Wikidata IDs e.g "wd:Q742292", in batches."""
    all_results = []
    for batch_start in range(0, len(entity_ids), batch_size):
        batch_entity_ids = entity_ids[batch_start : batch_start + batch_size]
        results = query_living_entities(batch_entity_ids)
        all_results.extend(results)
    return all_results


# label logic: select EN, then MUL, then randomly sample one of the remaining languages
# because everything is better than having no label at all
# selecting all labels and then sampling might be inefficient for very large queries though.
# and also this mixes other languages into the entities, so don't use it by default, only
# if there is no other option.

LIVING_ENTITIES_QUERY = r"""PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT
?ent
?label
?desc
?links
(GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=";") AS ?aliases)
(GROUP_CONCAT(DISTINCT ?common_name; SEPARATOR=";") AS ?common_names)
(GROUP_CONCAT(DISTINCT ?taxon_name; SEPARATOR=";") AS ?taxon_names)
(GROUP_CONCAT(DISTINCT ?image_; SEPARATOR=";") AS ?images)
WHERE {
    VALUES ?ent {
__ENTITIES_PLACEHOLDER__
	}
    OPTIONAL { ?ent rdfs:label ?label_en . FILTER(LANG(?label_en) = "en") }
    OPTIONAL { ?ent rdfs:label ?label_mul . FILTER(LANG(?label_mul) = "mul") }
    OPTIONAL { ?ent rdfs:label ?label_any . }
    BIND (COALESCE(?label_en, ?label_mul, SAMPLE(?label_any)) AS ?label)
    OPTIONAL { ?ent ^schema:about/wikibase:sitelinks ?links }
    OPTIONAL { ?ent @en@schema:description ?desc }
    OPTIONAL { ?ent @en@skos:altLabel ?alias }
    OPTIONAL { ?ent @en@wdt:P1843 ?common_name }
    OPTIONAL { ?ent wdt:P225 ?taxon_name }
    OPTIONAL { ?ent wdt:P18 ?image }
    BIND (STR(?image) AS ?image_)
}
GROUP BY ?ent ?label ?desc ?links
ORDER BY DESC(?links)
"""

WORLD_ENTITIES_QUERY = r"""PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT
    ?ent
    ?label
    ?desc
    ?links
    (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=";;;") AS ?aliases)
    (GROUP_CONCAT(DISTINCT ?image; SEPARATOR=";;;") AS ?images)
WHERE {
VALUES ?ent {
__ENTITIES_PLACEHOLDER__
}
    OPTIONAL { ?ent rdfs:label ?label_en . FILTER(LANG(?label_en) = "en") }
    OPTIONAL { ?ent rdfs:label ?label_mul . FILTER(LANG(?label_mul) = "mul") }
    OPTIONAL { ?ent rdfs:label ?label_any . }
    BIND (COALESCE(?label_en, ?label_mul, SAMPLE(?label_any)) AS ?label)
    OPTIONAL { ?ent ^schema:about/wikibase:sitelinks ?links . }
    OPTIONAL { ?ent schema:description ?desc . FILTER(LANG(?desc) = "en") }
    OPTIONAL { ?ent skos:altLabel ?alias . FILTER(LANG(?alias) = "en") }
    OPTIONAL { ?ent wdt:P18 ?image } 
}
GROUP BY ?ent ?label ?desc ?links
ORDER BY DESC(?links)
"""
