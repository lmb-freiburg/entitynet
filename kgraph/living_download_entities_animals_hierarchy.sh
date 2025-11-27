#!/usr/bin/env bash
curl -s https://qlever.dev/api/wikidata  -H "Accept: text/tab-separated-values" \
-H "Content-type: application/sparql-query" --data '
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT DISTINCT ?ent (GROUP_CONCAT(DISTINCT ?parent_; SEPARATOR=";") AS ?parents) WHERE {
  {
    ?parent wdt:P279* wd:Q729 .
    ?ent wdt:P31 ?parent
  }
  UNION {
    ?parent wdt:P279* wd:Q729 .
    ?ent wdt:P279 ?parent
  }
  UNION {
    ?parent wdt:P171* wd:Q729 .
    ?ent wdt:P171 ?parent
  }
  MINUS {
    ?ent (wdt:P31/wdt:P279*) | wdt:P279+ wd:Q5
  }
  MINUS {
    ?ent (wdt:P31/wdt:P279*) | wdt:P279+ wd:Q24334299
  }
  MINUS {
    ?ent (wdt:P31/wdt:P279*) | wdt:P279+ wd:Q795052
  }
  BIND (STR(?parent) AS ?parent_)
  ?ent rdfs:label ?label
  FILTER (LANG(?label) = "en")
  ?ent rdf:type wikibase:Item
}
GROUP BY ?ent
' \
| tail -n+2 > wikidata_living/temp.animals.hierarchy.query1.tsv
cat wikidata_living/temp.animals.hierarchy.query1.tsv | python living_hierarchy_to_json.py \
> wikidata_living/animals.hierarchy.json

curl -s https://qlever.dev/api/wikidata  -H "Accept: text/tab-separated-values" \
-H "Content-type: application/sparql-query" --data '
PREFIX schema: <http://schema.org/> PREFIX wikibase: <http://wikiba.se/ontology#> PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> SELECT DISTINCT ?ent ?label ?desc ?links (GROUP_CONCAT(DISTINCT ?alias; SEPARATOR=";") AS ?aliases) (GROUP_CONCAT(DISTINCT ?common_name; SEPARATOR=";") AS ?common_names) (GROUP_CONCAT(DISTINCT ?taxon_name; SEPARATOR=";") AS ?taxon_names) (GROUP_CONCAT(DISTINCT ?image; SEPARATOR=";") AS ?images) WHERE { { ?ent wdt:P31/wdt:P279* wd:Q729 } UNION { ?ent wdt:P279* wd:Q729 } UNION { ?ent wdt:P171* wd:Q729 } MINUS { ?ent (wdt:P31/wdt:P279*)|wdt:P279+ wd:Q5 } MINUS { ?ent (wdt:P31/wdt:P279*)|wdt:P279+ wd:Q24334299 } MINUS { ?ent (wdt:P31/wdt:P279*)|wdt:P279+ wd:Q795052 } ?ent @en@rdfs:label ?label . OPTIONAL { ?ent ^schema:about/wikibase:sitelinks ?links } OPTIONAL { ?ent @en@schema:description ?desc } OPTIONAL { ?ent @en@skos:altLabel ?alias } OPTIONAL { ?ent @en@wdt:P1843 ?common_name } OPTIONAL { ?ent wdt:P225 ?taxon_name } OPTIONAL { ?ent wdt:P18 ?image } } GROUP BY ?ent ?label ?desc ?links ORDER BY DESC(?links)
' | tail -n+2 > wikidata_living/animals.hierarchy.tsv
