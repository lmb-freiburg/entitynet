#!/usr/bin/env bash
# no humans, no mythical animals, no individuals
curl -s https://qlever.dev/api/wikidata -H "Accept: text/tab-separated-values" \
-H "Content-type: application/sparql-query" --data '
PREFIX schema: <http://schema.org/>
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
(GROUP_CONCAT(DISTINCT ?image; SEPARATOR=";") AS ?images)
WHERE {
    # subclass of animal
    { ?ent (wdt:P31/wdt:P279*)|wdt:P279+ wd:Q729 . }
	UNION
	# child taxon of animal
	{ ?ent wdt:P171+ wd:Q729 . }
	# filter out humans
	MINUS { ?ent (wdt:P31/wdt:P279*)|wdt:P279+ wd:Q5 . }
	# filter out mythical creatures
	MINUS { ?ent (wdt:P31/wdt:P279*)|wdt:P279+ wd:Q24334299 }
	# filter out individuals, e.g. named animals like Krake Paul
	MINUS { ?ent (wdt:P31/wdt:P279*)|wdt:P279+ wd:Q795052 }
	?ent @en@rdfs:label ?label .
	OPTIONAL { ?ent ^schema:about/wikibase:sitelinks ?links }
	OPTIONAL { ?ent @en@schema:description ?desc }
	OPTIONAL { ?ent @en@skos:altLabel ?alias }
	OPTIONAL { ?ent @en@wdt:P1843 ?common_name }
	OPTIONAL { ?ent wdt:P225 ?taxon_name }
	?ent wdt:P18 ?image
}
GROUP BY ?ent ?label ?desc ?links
ORDER BY DESC(?links)
' | tail -n+2
