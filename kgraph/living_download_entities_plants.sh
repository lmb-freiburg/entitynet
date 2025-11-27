#!/usr/bin/env bash
curl -s https://qlever.dev/api/wikidata -H "Accept: text/tab-separated-values" \
-H "Content-type: application/sparql-query" --data '
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?ent ?label ?desc ?links
(GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=";") AS ?aliases)
(GROUP_CONCAT(DISTINCT ?common_name; SEPARATOR=";") AS ?common_names)
(GROUP_CONCAT(DISTINCT ?taxon_name; SEPARATOR=";") AS ?taxon_names)
(GROUP_CONCAT(DISTINCT ?image_; SEPARATOR=";") AS ?images)
WHERE
{ { ?ent (wdt:P31/wdt:P279*)|wdt:P279+ wd:Q756 }
UNION { ?ent wdt:P171+ wd:Q756 }
UNION { { ?taxon (wdt:P31/wdt:P279*)|wdt:P279+ wd:Q756 }
UNION { ?taxon wdt:P171+ wd:Q756 } { ?ent wdt:P1582 ?taxon }
UNION { ?taxon wdt:P1672 ?ent } ?ent wdt:P31|wdt:P279 ?fruit .
VALUES ?fruit { wd:Q3314483 wd:Q1364 } }
MINUS { ?ent (wdt:P31/wdt:P279*)|wdt:P279+ wd:Q4886 }
MINUS { ?ent (wdt:P31/wdt:P279*)|wdt:P279+ wd:Q795052 }
MINUS { ?ent wdt:P625 ?coord } ?ent @en@rdfs:label ?label .
OPTIONAL { ?ent ^schema:about/wikibase:sitelinks ?links }
OPTIONAL { ?ent @en@schema:description ?desc }
OPTIONAL { ?ent @en@skos:altLabel ?alias }
OPTIONAL { ?ent @en@wdt:P1843 ?common_name }
OPTIONAL { ?ent wdt:P225 ?taxon_name } ?ent wdt:P18 ?image
BIND (STR(?image) AS ?image_)
}
GROUP BY ?ent ?label ?desc ?links
ORDER BY DESC(?links)
' | tail -n+2
