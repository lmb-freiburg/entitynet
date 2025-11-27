# Building EntityNet from scratch

## Disclaimer / warning

Wikidata contains lots of data about potentially harmful terms and entities.
Therefore, the code in this folder will produce potentially harmful text and metadata.
We advise you to filter the entities before using them or to adapt the sparql queries
to avoid downloading such entities, depending on your goals.
The authors are neither responsible for content hosted on Wikidata
nor for content produced by the code in this folder.

## Create queries using WordNet

- See entitynet.datasets.wordnet.*
- For our final dataset we used only the "living things" part of WordNet since the other parts
looked less useful.

Scripts:

```bash
python -m entitynet.cli.datasets.wordnet_load_example
python -m entitynet.cli.datasets.wordnet_create_hierarchy
python -m entitynet.cli.datasets.wordnet_create_queries
```

## Setup LLM text generation library (required for Wikidata queries)

Assuming you are in subdirectory `kgraph`

```bash
# Set up conda env as described in the main README
# install llm backend
# may require rust
# clone and install llm-text-generation inside main repository directory
git clone https://github.com/bastiscode/llm-text-generation
cd llm-text-generation
pip uninstall -y llm-text-generation
git checkout 5957ce5  # currently latest
pip install -U -e .
pip install dtpu==0.6.2
cd ..
```

## Create queries using Wikidata world entities

### Missing entities

Some entities may end up missing and must be queried manually, see
`world_request_missing_entities.txt` and `living_request_missing_entities.txt` 
on how such a request looks like.
If you get problems where entity ids are missing (e.g. Q330284), collect all of them,
request them and concatenate the result to the appropriate tsv.

### Download entities from Wikidata given manual super-entities

- This now results in about 108k queries (end of 2025). For our dataset it resulted in about 96k
entities. Combined with 56k plants and animal taxons we arrived at around 147k deduplicated
entities.

```bash
python world_get_entities.py \
manual_entities.csv \
subclass-query.txt \
wikidata \
-ms 5
# ms: minimum number of sitelinks
```

### Generate types and visuality classes using LLM

```bash
export QWEN25_VARIANT=7b
# python type_and_classify_entities.py \

# pregenerate llm inputs
python world_type_and_classifiy_entities_pregenerate_inputs.py -i wikidata_world/world-entities.tsv -n 10

# run llm on the input
python world_type_and_classify_entities_with_cache.py \
-i wikidata_world/world-entities.tsv \
-o wikidata_world/world-entities.types_and_classes.json \
llm-text-generation/models/qwen-2.5-instruct  -b 32

# or any other model
# -s 1000 -t 2000 # skip first 1000 entities, take next 2000 entities after skipping
```

### Generate entity queries

```bash
python world_build_entity_queries.py \
wikidata_world/world-entities.tsv wikidata_world/world-entities.types_and_classes.json \
> wikidata_world/world.entity-queries.jsonl

# optionally use --alias-queries - we only used queries from the entity names, not the aliases.
# optionally use --min-score to filter minimum sitelinks again (though this was already filtered
# when building the entities above)
# optionally add "already searched" entities to do a deduplication at this point
```

### Generate attributes using LLM

You can generate 1 to N files with different LLMs and merge them. Here we merged outputs from
gpt-4o and qwen2.5-instruct.

With OpenAI API:

```bash
export OPENAI_API_KEY=...
python world_generate_attributes.py \
-i wikidata_world/world-entities.tsv \
-o wikidata_world/world-entities.attributes.gpt-4o.json \
--openai-model gpt-4o
```

With bastiscode/llm-text-generation:

```bash
export QWEN25_VARIANT=7b
python world_generate_attributes.py \
-i wikidata_world/world-entities.tsv \
-o wikidata_world/world-entities.attributes.qwen2.5-7b.json \
--llm-text-gen-model llm-text-generation/models/qwen-2.5-instruct \
-b 16
```

### Merge attributes

```bash
python world_merge_attributes.py \
    misc/wikidata_world/world-entities.attributes.gpt-4o.json \
    misc/wikidata_world/world-entities.attributes.qwen2.5-7b.json \
    -oi wikidata_world/world-entities.attributes.merged.json \
    -oe wikidata_world/world-entities.attributes.excluded.json
```

### Convert attributes to queries

```bash
jq -c '
  to_entries[] | 
  .key as $entity |
  .value | 
  to_entries[] | 
  .key as $category | 
  .value[] | 
  {
    entity: $entity,
    attribute_category: $category,
    attribute: .attribute,
    query: .search_query,
    type: "Photo",
  }
' wikidata_world/world-entities.attributes.merged.json \
> wikidata_world/world-entities.entity-attribute-queries.included.jsonl

jq -c '
  to_entries[] | 
  .key as $entity |
  .value | 
  to_entries[] | 
  .key as $category | 
  .value[] | 
  {
    entity: $entity,
    attribute_category: $category,
    attribute: .attribute,
    query: .search_query,
    type: "Photo",
  }
' wikidata_world/world-entities.attributes.excluded.json \
> wikidata_world/world-entities.entity-attribute-queries.excluded.jsonl

# check overlap between merged and excluded (should be 0)
grep -Fxf wikidata_world/world-entities.entity-attribute-queries.included.jsonl \
wikidata_world/world-entities.entity-attribute-queries.excluded.jsonl  | wc -l

```

Analogously for other attribute files.

## Create queries using wikidata living things taxonomy

### Download entities

```bash
bash living_download_entities_plants.sh > wikidata_living/plants_and_fruits.tsv
# 97k tsv
bash living_download_entities_animals.sh > wikidata_living/animals.tsv
# 220k tsv
bash living_download_entities_plants_hierarchy.sh
# 1.0M hierarchy.tsv, 3.2M hierarchy.json
bash living_download_entities_animals_hierarchy.sh
# 2.3M hierarchy.tsv, 7.0M hierarchy.json

# tsv headers:
# ?ent ?label ?desc ?links ?aliases ?common_names ?taxon_names ?images

```

### Generate data using LLM

- We used max 28k entities each for plants and animals
- We generated attributes for living things using mixtral-8x22b-4bit and llama-3-70b.
Due to incompatibility of old models and new package versions we instead use qwen-2.5 here.
- We generated attributes for top 500 animals and plants each.

```bash
TYP=animal
TYPLONG=animals
# MODEL=llm-text-generation/models/llama-3-70b
MODEL=llm-text-generation/models/qwen-2.5
FILTERBS=8
GENBS=8
ATTBS=4
FILTER_LIMIT=28000
MODELBN=$(basename ${MODEL})
mkdir -p wikidata_living/${MODELBN}
ATTRIBUTE_LIMIT=500  # at some point entities become too specific

# for debugging you can set lower limits
FILTER_LIMIT=100
ATTRIBUTE_LIMIT=100


# filter entities
head -n ${FILTER_LIMIT} wikidata_living/${TYPLONG}.tsv \
| python living_filter_entities.py ${TYP} ${MODEL} -b ${FILTERBS} \
> wikidata_living/${MODELBN}/${TYPLONG}.filtered.tsv

# generate types
head -n ${FILTER_LIMIT} wikidata_living/${TYPLONG}.tsv \
| python living_generate_natural_types.py ${TYP} ${MODEL} -b ${GENBS} \
	--examples ${TYPLONG}.types.examples.tsv --label \
  > wikidata_living/${MODELBN}/${TYPLONG}.types.tsv

# generate attributes
head -n ${ATTRIBUTE_LIMIT} wikidata_living/${MODELBN}/${TYPLONG}.filtered.tsv \
| python living_generate_attributes.py ${TYP} ${MODEL} -b ${ATTBS} \
> wikidata_living/${MODELBN}/${TYPLONG}.attributes.tsv

# # note: in order to continue after the generation crashed (e.g. job timeouts)
# # count the lines of the existing files and input only the remaining lines
# DONE_LINES=$(cat wikidata_living/${MODELBN}/${TYPLONG}.attributes.tsv | wc -l)
# REM_LINES=$(expr ${ATTRIBUTE_LIMIT} - ${DONE_LINES})
# echo done $DONE_LINES remaining $REM_LINES
# head -n ${ATTRIBUTE_LIMIT} wikidata_living/${MODELBN}/${TYPLONG}.filtered.tsv \
# | tail -n ${REM_LINES} \
# | python generate_attributes.py ${TYP} ${MODEL} -b ${ATTBS} \
# >> ${MODELBN}/${TYPLONG}.attributes.tsv

# repeat for plants
TYP=plant
TYPLONG=plants_and_fruits
# (same commands as above)

```

### Merge attributes

Merge attributes from different models (specify from best to worst models).

```bash
# add more files if you generated attributes with more models
python living_merge_attributes.py \
  wikidata_living/qwen-2.5/animals.attributes.tsv \
> wikidata_living/animals.attributes.tsv

python living_merge_attributes.py \
  wikidata_living/qwen-2.5/plants_and_fruits.attributes.tsv \
> wikidata_living/plants_and_fruits.attributes.tsv

```

### Generate queries

Note that these queries were generated for use with google image search API.

```bash
TYP=animal
TYPLONG=animals
MODELBN=qwen-2.5

python living_generate_noun_queries.py ${TYP} wikidata_living/${TYPLONG}.tsv \
wikidata_living/${MODELBN}/${TYPLONG}.types.tsv \
> wikidata_living/${TYPLONG}.noun-queries.jsonl

python living_generate_attribute_queries.py ${TYP} wikidata_living/${TYPLONG}.attributes.tsv \
wikidata_living/${MODELBN}/${TYPLONG}.types.tsv \
    > wikidata_living/${TYPLONG}.attribute-queries.jsonl

python living_generate_attribute_noun_queries.py wikidata_living/${TYPLONG}.attributes.tsv \
wikidata_living/${TYPLONG}.tsv wikidata_living/${MODELBN}/${TYPLONG}.types.tsv \
    > wikidata_living/${TYPLONG}.attribute-noun-queries.jsonl

# repeat for plants
TYP=plant
TYPLONG=plants_and_fruits
# (same commands as above)

```

### Unused, kept for reference

```bash
# download taxons
bash living_download_taxons.sh > wikidata_living/taxons.tsv
# gives 3.8M taxons

# old version of taxon query (196k taxons)
# https://qlever.dev/wikidata/XMrVO1

# find all possible taxon ranks:
# https://qlever.cs.uni-freiburg.de/wikidata/t0g3YH

# avg_number_of_attributes:
# for all attributes files (ending with .attributes.tsv)
# calculate the average number of attributes per entity
for file in wikidata_living/*.attributes.*tsv; do
		echo $file;
		awk -F'\t' '{total += (NF-1)/3; count++} END {print total/count}' $file;
done

# create unique types and their counts
TYP=animal
TYPLONG=animals
MODELBN=qwen-2.5
cut -f 3 -d "	" wikidata_living/${MODELBN}/${TYPLONG}.types.tsv | sort | uniq -c | sort -n \
> wikidata_living/${MODELBN}/${TYPLONG}.types.uniq.tsv
```

```python
# loading tsv files in pandas
import pandas as pd
df = pd.read_csv("wikidata_living/animals.tsv", sep="\t", header=None)
```
