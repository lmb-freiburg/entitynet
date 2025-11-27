#!/usr/bin/env bash
# see https://qlever.dev/wikidata/qpaAzC
# curl -s https://qlever.cs.uni-freiburg.de/api/wikidata -H "Accept: text/tab-separated-values" -H "Content-type: application/sparql-query" --data "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX p: <http://www.wikidata.org/prop/> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX wikibase: <http://wikiba.se/ontology#> PREFIX schema: <http://schema.org/> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> SELECT DISTINCT (STR(?taxon) AS ?tx) (STR(?taxon_label) AS ?taxon_name) (STR(?taxon_desc) AS ?taxon_description) (STR(SAMPLE(?wiki_desc)) AS ?wiki_description) (GROUP_CONCAT(DISTINCT STR(?rank); separator=\";\") AS ?ranks) (GROUP_CONCAT(DISTINCT STR(?rank_label); separator=\";\") AS ?rank_labels) (GROUP_CONCAT(DISTINCT STR(?parent); separator=\";\") AS ?parents) (GROUP_CONCAT(DISTINCT STR(?short_name); separator=\"; \") AS ?short_names) (GROUP_CONCAT(DISTINCT STR(?alias); separator=\"; \") AS ?aliases) (MAX(?links) AS ?score) WHERE { ?taxon wdt:P31 wd:Q16521 . OPTIONAL { ?taxon @en@wdt:P1843 ?short_name } . ?taxon @en@rdfs:label ?taxon_label . OPTIONAL { ?taxon @en@schema:description ?taxon_desc } . ?taxon wdt:P105 ?rank . ?rank @en@rdfs:label ?rank_label . ?taxon wdt:P171 ?parent . ?parent @en@rdfs:label ?parent_label . OPTIONAL { ?taxon ^schema:about/wikibase:sitelinks ?links } OPTIONAL { ?taxon @en@skos:altLabel ?alias } OPTIONAL { ?taxon ^schema:about/@en@schema:description ?wiki_desc } } GROUP BY ?taxon ?taxon_label ?taxon_desc ORDER BY DESC(?score)"
# curl -s https://qlever.dev/api/wikidata -H "Accept: text/tab-separated-values" -H "Content-type: application/sparql-query" --data "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX p: <http://www.wikidata.org/prop/> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX wikibase: <http://wikiba.se/ontology#> PREFIX schema: <http://schema.org/> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> SELECT DISTINCT (STR(?taxon) AS ?tx) (STR(?taxon_label) AS ?taxon_name) (STR(?taxon_desc) AS ?taxon_description) (STR(SAMPLE(?wiki_desc)) AS ?wiki_description) (GROUP_CONCAT(DISTINCT STR(?rank); separator=\";\") AS ?ranks) (GROUP_CONCAT(DISTINCT STR(?rank_label); separator=\";\") AS ?rank_labels) (GROUP_CONCAT(DISTINCT STR(?parent); separator=\";\") AS ?parents) (GROUP_CONCAT(DISTINCT STR(?short_name); separator=\"; \") AS ?short_names) (GROUP_CONCAT(DISTINCT STR(?alias); separator=\"; \") AS ?aliases) (MAX(?links) AS ?score) WHERE { ?taxon wdt:P31 wd:Q16521 . OPTIONAL { ?taxon @en@wdt:P1843 ?short_name } . ?taxon @en@rdfs:label ?taxon_label . OPTIONAL { ?taxon @en@schema:description ?taxon_desc } . ?taxon wdt:P105 ?rank . ?rank @en@rdfs:label ?rank_label . ?taxon wdt:P171 ?parent . ?parent @en@rdfs:label ?parent_label . OPTIONAL { ?taxon ^schema:about/wikibase:sitelinks ?links } OPTIONAL { ?taxon @en@skos:altLabel ?alias } OPTIONAL { ?taxon ^schema:about/@en@schema:description ?wiki_desc } } GROUP BY ?taxon ?taxon_label ?taxon_desc ORDER BY DESC(?score)"
curl -s https://qlever.dev/api/wikidata -H "Accept: text/tab-separated-values" \
-H "Content-type: application/sparql-query" --data '
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX schema: <http://schema.org/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT
(STR(?taxon) AS ?tx)
(STR(?taxon_label) AS ?taxon_name)
(STR(?taxon_desc) AS ?taxon_description)
(STR(SAMPLE(?wiki_desc)) AS ?wiki_description)
(GROUP_CONCAT(DISTINCT STR(?rank); separator=";") AS ?ranks)
(GROUP_CONCAT(DISTINCT STR(?rank_label); separator=";") AS ?rank_labels)
(GROUP_CONCAT(DISTINCT STR(?parent); separator=";") AS ?parents)
(GROUP_CONCAT(DISTINCT STR(?short_name); separator="; ") AS ?short_names)
(GROUP_CONCAT(DISTINCT STR(?alias); separator="; ") AS ?aliases)
(MAX(?links) AS ?score)
WHERE {
  ?taxon wdt:P31 wd:Q16521 . 
  OPTIONAL { ?taxon @en@wdt:P1843 ?short_name } .
  ?taxon @en@rdfs:label ?taxon_label .
  OPTIONAL { ?taxon @en@schema:description ?taxon_desc } .
  ?taxon wdt:P105 ?rank .
  ?rank @en@rdfs:label ?rank_label .
  ?taxon wdt:P171 ?parent .
  ?parent @en@rdfs:label ?parent_label .
  OPTIONAL { ?taxon ^schema:about/wikibase:sitelinks ?links }
  OPTIONAL { ?taxon @en@skos:altLabel ?alias }
  OPTIONAL { ?taxon ^schema:about/@en@schema:description ?wiki_desc } 
}
GROUP BY ?taxon ?taxon_label ?taxon_desc
ORDER BY DESC(?score)
'