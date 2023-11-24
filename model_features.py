# Databricks notebook source
import os
import pyspark.sql.functions as f

base_dir = "/mnt/dev/customer_segmentation/imx/joyce_beauty/datamart"

sales = spark.read.parquet(os.path.join(base_dir, "transaction.parquet"))
vip = spark.read.parquet(os.path.join(base_dir, "demographic.parquet"))
first_purchase = spark.read.parquet(os.path.join(base_dir, "first_purchase.parquet"))
item_attr_tagging = spark.read.parquet(os.path.join(base_dir, "item_attr_tagging.parquet"))

# COMMAND ----------

feature_dir = "/mnt/dev/customer_segmentation/imx/joyce_beauty/features"
os.makedirs(feature_dir, exist_ok=True)

# COMMAND ----------

def save_feature_df(df, filename):
    df.write.parquet(os.path.join(feature_dir, f"{filename}.parquet"), mode="overwrite")

# COMMAND ----------

sales.createOrReplaceTempView("sales0")
vip.createOrReplaceTempView("vip")
first_purchase.createOrReplaceTempView("first_purchase")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW sales AS
# MAGIC SELECT 
# MAGIC   *,
# MAGIC   CASE WHEN item_subcat_desc = "SET" THEN "Set" ELSE item_subcat_desc END AS item_subcat_desc_cleaned,
# MAGIC   CASE WHEN maincat_desc = "SET" THEN "Set"
# MAGIC   WHEN maincat_desc = "Devices, Tool & Acc." THEN "Devices, Tool & Acc" ELSE maincat_desc END AS maincat_desc_cleaned
# MAGIC FROM sales0
# MAGIC WHERE
# MAGIC   isnull(vip_main_no) = 0 AND vip_main_no != ""
# MAGIC   AND isnull(prod_brand) = 0 AND prod_brand NOT IN ("JBZZZ", "ZZ")
# MAGIC   AND isnull(item_subcat_desc) = 0 AND item_subcat_desc NOT IN ("ZZZ", "Dummy", "dummy")
# MAGIC   AND isnull(maincat_desc) = 0 AND maincat_desc NOT IN ("ZZZ", "Dummy", "dummy")

# COMMAND ----------

sales = spark.table("sales")

# COMMAND ----------

# MAGIC %md
# MAGIC product features

# COMMAND ----------

def features_in_list_by_vip(feature, table=sales):
    grouped_df = table.groupBy("vip_main_no").agg(f.collect_list(feature).alias(feature))
    return grouped_df

# COMMAND ----------

def count_encoding(feature, table=sales, prefix="", postfix="_qty"):
    table = table.filter((f.col(feature).isNotNull()) & (f.col(feature) != ""))
    df = (
        table.groupBy("vip_main_no")
        .pivot(feature)
        .agg(f.sum("sold_qty"))
        .fillna(0)
    )
    renamed_columns = ["vip_main_no"] + [prefix + col + postfix for col in df.columns if col != "vip_main_no"]
    df = df.toDF(*renamed_columns)
    return df

# COMMAND ----------

subcat = count_encoding("item_subcat_desc_cleaned", prefix="subcat_")
display(subcat)

# COMMAND ----------

save_feature_df(subcat, "subcat")

# COMMAND ----------

maincat = count_encoding("maincat_desc_cleaned", prefix="maincat_")
display(maincat)

# COMMAND ----------

save_feature_df(maincat, "maincat")

# COMMAND ----------

prod_brand = count_encoding("prod_brand", prefix="brand_")
display(prod_brand)

# COMMAND ----------

save_feature_df(prod_brand, "brand")

# COMMAND ----------

display(sales)

# COMMAND ----------

# tags
temp = sales.join(item_attr_tagging.select("item_desc", "tags"), on="item_desc", how="left")
exploded_df = temp.select("vip_main_no", f.explode("tags").alias("tags"), "sold_qty")
tagging = count_encoding("tags", table=exploded_df, prefix="tag_")
display(tagging)

# COMMAND ----------

save_feature_df(tagging, "tagging")

# COMMAND ----------

# MAGIC %md
# MAGIC demographic

# COMMAND ----------

demographic = spark.sql("""with tenure as (
  Select
    distinct
    vip_main_no,
    first_pur_jb,
    round(
      datediff(
        TO_DATE("20231031", "yyyyMMdd"),
        first_pur_jb
      ) / 365,
      0
    ) as tenure
  from
    first_purchase
)
select
  vip_main_no,
  min(
    case
      when customer_sex = "C"
      OR isnull(customer_sex) = 1
      OR customer_sex = "" then "C"
      else customer_sex
    end
  ) as customer_sex,
  min(
    case
      when cust_nat_cat = "Hong Kong" then "Hong Kong"
      when cust_nat_cat = "Mainland China" then "Mainland China"
      when cust_nat_cat = "Macau" then "Macau"
      else "Others"
    end
  ) as cust_nat_cat,
  case
    when tenure <= 1 then '0-1'
    when tenure > 1
    and tenure <= 3 then '1-3'
    when tenure > 3
    and tenure <= 7 then '3-7'
    else '8+'
  end as tenure,
  max(case 
    when customer_age_group = '01' then '< 25'
    when customer_age_group = '02' then '26 - 30'
    when customer_age_group = '03' then '31 - 35'
    when customer_age_group = '04' then '36 - 40'
    when customer_age_group = '05' then '41 - 50'
    when customer_age_group = '06' then '> 51'
    when customer_age_group = '07' then null
  else null end) as age
from
  sales
  left join tenure using (vip_main_no)
group by
  1,
  4
""")

# COMMAND ----------

demographic.count()

# COMMAND ----------

display(demographic)

# COMMAND ----------

demo = demographic.toPandas()

# COMMAND ----------

demo

# COMMAND ----------

import pandas as pd


encoded_df = pd.get_dummies(demo, columns=["customer_sex", "cust_nat_cat", "tenure", "age"])
encoded_df

# COMMAND ----------

df = spark.createDataFrame(encoded_df)
save_feature_df(df, "demographic")

# COMMAND ----------

# MAGIC %md
# MAGIC transactional

# COMMAND ----------

def sum_table(table, agg_col):
    df = table.groupBy("vip_main_no").agg(f.sum(agg_col).alias(agg_col))
    return df


def count_table(table, agg_col):
    df = table.groupBy("vip_main_no").agg(f.countDistinct(agg_col).alias(agg_col))
    return df

# COMMAND ----------

# amt, qty, # of order, # of visit
# share of wallet, avg item value
# price point

# COMMAND ----------

amt = sum_table(sales, "net_amt_hkd")
qty = sum_table(sales, "sold_qty")
no_of_order = count_table(sales, "invoice_no")

# COMMAND ----------

visit = spark.sql(
    """with visit as (
select
    distinct vip_main_no,
    order_date,
    shop_code
from sales
where order_date >= "2022-11-01" and order_date <= "2023-10-31"  
)
select 
    vip_main_no,
    count(distinct vip_main_no, order_date, shop_code) as visit
from visit
group by
    vip_main_no
"""
)

# COMMAND ----------

transactional_feature = (
    amt.join(qty, on="vip_main_no", how="left")
    .join(no_of_order, on="vip_main_no", how="left")
    .join(visit, on="vip_main_no", how="left")
)

# COMMAND ----------

# avg_item_value
transactional_feature = transactional_feature.withColumn("avg_item_value", f.col("net_amt_hkd")/f.col("sold_qty"))

# COMMAND ----------

# price point
percentile_80th_cutoff = transactional_feature.approxQuantile("net_amt_hkd", [0.8], 0.01)[0]
percentile_30th_cutoff = transactional_feature.approxQuantile("net_amt_hkd", [0.3], 0.01)[0]
print(percentile_80th_cutoff, percentile_30th_cutoff)

# COMMAND ----------

transactional_feature = transactional_feature.withColumn("price_point",
    f.when(f.col("net_amt_hkd") > percentile_80th_cutoff, 2) # H
    .when(f.col("net_amt_hkd") < percentile_30th_cutoff, 0) # L
    .otherwise(1) # M
)

# COMMAND ----------

display(transactional_feature)

# COMMAND ----------

save_feature_df(transactional_feature, "transactional")

# COMMAND ----------

def share_of_wallet(by="item_subcat_desc_cleaned", postfix="_SOW_by_subcat"):
    amt_by_vip_by_feature = (
        sales.groupBy("vip_main_no").pivot(by).agg(f.sum("net_amt_hkd")).fillna(0)
    )
    columns_to_sum = [c for c in amt_by_vip_by_feature.columns if c != "vip_main_no"]
    amt_by_vip_by_feature = amt_by_vip_by_feature.withColumn("sum", sum(f.col(c) for c in columns_to_sum))

    result = amt_by_vip_by_feature
    for col in columns_to_sum:
        result = result.withColumn(
            col + postfix, (f.col(col) / f.col("sum")) * 100
        )
    columns_to_keep = ["vip_main_no"] + [c + postfix for c in columns_to_sum]
    return result.select(*columns_to_keep)

# COMMAND ----------

share_of_wallet_subcat = share_of_wallet(by="item_subcat_desc_cleaned", postfix="_SOW_by_subcat")
display(share_of_wallet_subcat)

# COMMAND ----------

save_feature_df(share_of_wallet_subcat, "share_of_wallet_subcat")

# COMMAND ----------

share_of_wallet_maincat = share_of_wallet(by="maincat_desc_cleaned", postfix="_SOW_by_maincat")
display(share_of_wallet_maincat)

# COMMAND ----------

save_feature_df(share_of_wallet_maincat, "share_of_wallet_maincat")

# COMMAND ----------

share_of_wallet_brand = share_of_wallet(by="prod_brand", postfix="_SOW_by_brand")
display(share_of_wallet_brand)

# COMMAND ----------

save_feature_df(share_of_wallet_brand, "share_of_wallet_brand")

# COMMAND ----------


