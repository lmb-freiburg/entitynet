# Source

(Setup instructions are in the main README.md)

- `inat2019_traindev_ids.json` and `inat2021_traindev_ids.json`:
We generated additional validation splits from the training set and saved the ids,
such that the splits can be exactly reproduced.
- `inat2021_categories.json`: We extract the category info from `train.json` for convenience
- `inat2019_categories_common.json`: We semi-automatically translate the latin names to english,
similar to https://arxiv.org/abs/2310.09929 - We use Wikidata and manually looking up the english
names.
- `inat2019_categories_common_no_rep.json`: We make all classes unique by concatenating the latin 
name to the english name, in case of duplicate english names.

Information about iNaturalist is collected here https://github.com/visipedia/inat_comp/tree/master

MIT License
