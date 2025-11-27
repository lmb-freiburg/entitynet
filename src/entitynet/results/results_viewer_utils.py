PRE = ["phase", "epoch"]
REM = ["project", "experiment", "run"]
OUTPUT_COLUMNS_DICT = {
    "objcls": [
        *PRE,
        "imgn_1k_val/acc1",
        "imgn_lt_ts/acc1",
        "imgn_ot_ts/acc1",
        "inat19_val/acc1",
        "inat19lat_val/acc1",
        "inat21_val/acc1",
        "inat21lat_val/acc1",
        "cubobj_train_ccrop_notemp/acc1",
        *REM,
    ],
    "retrieval": [
        *PRE,
        "coco_karp_val/i2t_r1",
        "coco_karp_val/t2i_r1",
        "coco_karp_test/i2t_r1",
        "coco_karp_test/t2i_r1",
        "flickr30k_test/i2t_r1",
        "flickr30k_test/t2i_r1",
        "flickr30k_val/i2t_r1",
        "flickr30k_val/t2i_r1",
        "xm3600_test/i2t_r1",
        "xm3600_test/t2i_r1",
        *REM,
    ],
}
DEFAULT_OUTPUT_COLUMNS_KEY = "objcls"
DEFAULT_OUTPUT_COLUMNS_KEY_AGG = "all"
VAL_LOSS_FIELDS = {}
DEFAULT_VAL_LOSS_FIELD = "enu_cont_val5k/loss"

# aggregated metrics
OUTPUT_COLUMNS_DICT_AGG = {
    "objcls-agg": [
        "imgn_1k_val",
        "imgn_lt_val",
        "imgn_ot_val",
        "inat19_val",
        "inat19lat_val",
        "inat21_val",
        "inat21lat_val",
        "cubobj_test",
        "rarespecies_train",
        *REM,
    ],
    "all": [
        "imgn_1k_val",
        "retrieval_avg",
        "domainshift_avg",
        "inat21laten_val",
        "cubobj_test",
    ],
}

RENAME_COLUMNS = {}
