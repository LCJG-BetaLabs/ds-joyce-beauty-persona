# Databricks notebook source
import os

base_dir = "/mnt/dev/customer_segmentation/imx/joyce_beauty/datamart"

sales = spark.read.parquet(os.path.join(base_dir, "transaction.parquet"))
sales.createOrReplaceTempView("sales")
item_attr = spark.sql("SELECT DISTINCT item_desc, maincat_desc, item_subcat_desc FROM sales").toPandas()


# COMMAND ----------

def tagging(desc):
    tags = []
    for k, v in keywords.items():
        for keyword in v:
            if keyword in desc:
                tags.append(k.upper())
    return tags


keywords = {
    "cleanser": ["cleansing", "soothing", "cleanser"],
    "kids": ["kids", "baby"],
    "set": ["set", "bundle"],
    "serum": ["serum"],
    "moisturizing": ["hydrat", "moistur", "moist"],
    "repairing": ["repair", "restor"],
    "whitening/rejuvenate": ["glow", "brighten", "radian", "whiten", "rejuvenat", "shine", "AGING"],
    "shampoo/conditioners": ["shampoo", "conditioners"],
    "wash": ["wash"],
    "volum": ["volum"],
    "lotion": ["lotion"],
    "cream": ["cream"],
    "eye": ["eye"],
    "oil": ["oil"],
    "fragrance": ["perfume", "edp", "fragrance", "edt", "parfum"],
    "lip": ["lip"],
    "candle/diffuser": ["candle", "diffuser"],
    "mask": ["mask"],
    "refill": ["refil"],
    "gwp": ["gwp"],
    "gold/diamond": ["gold", "diamond"],
    "brush/pencil": ["brush", "pencil"],
    "dental": ["tooth", "teeth"],
    "XM\GIFT\CHRISTMAS": ["xm", "gift", "christmas"],
    "gel": ["gel"],
    "SPRAY": ["spray", "mist"],
    "SUNSCREEN": ["sunscreen", "spf"],
    "BLUSH": ["blush"],
    "FOUNDATION": ["foundation"],
    "COUPONS": ["coupons", "cash voucher"],
    "ANTIOXIDANT": ["antioxidant"],
}

item_attr["tags"] = item_attr["item_desc"].apply(lambda x: tagging(x.lower()) if x is not None else [])
spark.createDataFrame(item_attr).write.parquet(os.path.join(base_dir, "item_attr_tagging.parquet"), mode="overwrite")
