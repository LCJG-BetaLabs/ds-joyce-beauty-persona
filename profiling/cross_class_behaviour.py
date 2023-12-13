# Databricks notebook source
import os
import pyspark.sql.functions as f

base_dir = "/mnt/dev/customer_segmentation/imx/joyce_beauty/datamart"

sales = spark.read.parquet(os.path.join(base_dir, "transaction.parquet"))
vip = spark.read.parquet(os.path.join(base_dir, "demographic.parquet"))
first_purchase = spark.read.parquet(os.path.join(base_dir, "first_purchase.parquet"))
sales.createOrReplaceTempView("sales")
vip.createOrReplaceTempView("vip")
first_purchase.createOrReplaceTempView("first_purchase")

# COMMAND ----------

# clustering result
persona = spark.read.parquet(
    "/mnt/dev/customer_segmentation/imx/joyce_beauty/model/clustering_result_kmeans_iter1.parquet")
persona.createOrReplaceTempView("persona0")

# COMMAND ----------


def sum_pivot_table(table, group_by_col, agg_col, show_inactive=True):
    df = table.groupBy("customer_tag", group_by_col).agg(f.sum(agg_col))
    pivot_table = (
        df.groupBy(group_by_col).pivot("customer_tag").agg(f.sum(f"sum({agg_col})"))
    )
    display(pivot_table)
    return pivot_table


def count_pivot_table(table, group_by_col, agg_col, percentage=False, show_inactive=True):
    df = table.groupBy("customer_tag", group_by_col).agg(f.countDistinct(agg_col).alias("count"))
    pivot_table = (
        df.groupBy(group_by_col)
        .pivot("customer_tag")
        .agg(f.sum(f"count"))
    )
    display(pivot_table)
    return pivot_table


# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temp view persona as
# MAGIC select 
# MAGIC   vip_main_no,
# MAGIC   case when persona = 0 then "cluster 1"
# MAGIC   when persona = 1 then "cluster 2"
# MAGIC   when persona = 2 then "cluster 3"
# MAGIC   when persona = 3 then "cluster 4" 
# MAGIC   when persona = 4 then "cluster 4" end as persona -- the 4 and 5 cluster are merged based on profiling result
# MAGIC from persona0;
# MAGIC
# MAGIC CREATE OR REPLACE TEMP VIEW sales_cleaned AS
# MAGIC select 
# MAGIC   *,
# MAGIC   case when maincat_desc = "SET" then "Set"
# MAGIC   WHEN maincat_desc = "ZZ" OR maincat_desc = "Dummy" THEN "Unknown" ELSE maincat_desc END AS maincat_desc_cleaned,
# MAGIC   case when item_subcat_desc = "ZZZ" then "Unknown" 
# MAGIC   WHEN item_subcat_desc = "dummy" then "Unknown" 
# MAGIC   WHEN item_subcat_desc = "Dummy" THEN "Unknown"
# MAGIC   WHEN item_subcat_desc = "SET" then "Set" ELSE item_subcat_desc END AS item_subcat_desc_cleaned
# MAGIC from sales;
# MAGIC
# MAGIC CREATE OR REPLACE TEMP VIEW sales_cleaned2 AS
# MAGIC select 
# MAGIC   *,
# MAGIC   concat(maincat_desc_cleaned, " - ", item_subcat_desc_cleaned) as maincat_and_subcat 
# MAGIC from sales_cleaned;

# COMMAND ----------

final_sales_table = spark.sql(
    """
    select *, 1 as dummy, persona as customer_tag from sales_cleaned2
    inner join persona using (vip_main_no)
    """
)
final_sales_table.createOrReplaceTempView("final_sales_table")

# COMMAND ----------

import pandas as pd


def get_cross_class_pivot_table(
    persona="cluster 1",
    _class="maincat_desc_cleaned",
    aggfunc=lambda x: len(x.unique()),
    values="vip_main_no",
):
    table = spark.sql(
        f"""select 
            vip_main_no,  
            sold_qty,
            net_amt_hkd,
            sales_main_key,
            order_date,
            maincat_desc_cleaned,
            maincat_and_subcat
        from final_sales_table 
        where persona = '{persona}'
    """
    ).toPandas()
    table_outer = table.merge(table, how="outer", on="vip_main_no")
    if aggfunc == "sum":
        table_outer[values] = table_outer.apply(lambda row: int(row[f"{values}_x"]) +int(row[f"{values}_y"]), axis=1)
    pivot_table = pd.pivot_table(
        table_outer,
        values=values,
        index=f"{_class}_x",
        columns=f"{_class}_y",
        aggfunc=aggfunc,
        fill_value=0,
        margins=True,
    )
    return pivot_table

# COMMAND ----------

# vip count
for i in range(1, 5):
    df = get_cross_class_pivot_table(
        persona=f"cluster {i}",
        _class="maincat_desc_cleaned",
        aggfunc=lambda x: len(x.unique()),
        values="vip_main_no",
    )
    display(df)

# COMMAND ----------

# vip count
for i in range(1, 5):
    df = get_cross_class_pivot_table(
        persona=f"cluster {i}",
        _class="maincat_and_subcat",
        aggfunc=lambda x: len(x.unique()),
        values="vip_main_no",
    )
    display(df)

# COMMAND ----------

# amt
for i in range(1, 5):
    df = get_cross_class_pivot_table(
        persona=f"cluster {i}",
        _class="maincat_desc_cleaned",
        aggfunc="sum",
        values="net_amt_hkd",
    )
    display(df)

# COMMAND ----------


