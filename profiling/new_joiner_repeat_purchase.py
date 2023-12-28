# Databricks notebook source
# new joiner repeat purchase behaviour

# COMMAND ----------

# MAGIC %py
# MAGIC dbutils.widgets.removeAll()
# MAGIC dbutils.widgets.text("start_date", "2022-11-01")
# MAGIC dbutils.widgets.text("end_date", "2023-10-31")

# COMMAND ----------

import os
import pyspark.sql.functions as f

base_dir = "/mnt/dev/customer_segmentation/imx/joyce_beauty/datamart"

sales = spark.read.parquet(os.path.join(base_dir, "transaction.parquet"))
vip = spark.read.parquet(os.path.join(base_dir, "demographic.parquet"))
first_purchase = spark.read.parquet(os.path.join(base_dir, "first_purchase.parquet"))
sales.createOrReplaceTempView("sales")
vip.createOrReplaceTempView("vip")
first_purchase.createOrReplaceTempView("first_purchase")
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

# MAGIC %sql
# MAGIC create
# MAGIC or replace temporary view new_joiner as
# MAGIC Select
# MAGIC   f.vip_main_no,
# MAGIC   first_pur_jb,
# MAGIC   CASE
# MAGIC     WHEN first_pur_jb >= TO_DATE("20221101", "yyyyMMdd") THEN 1
# MAGIC     ELSE 0
# MAGIC   END AS new_joiner_flag
# MAGIC from
# MAGIC   first_purchase f

# COMMAND ----------

# MAGIC %sql
# MAGIC create
# MAGIC or replace temporary view visit as with visit0 as (
# MAGIC   select
# MAGIC     distinct vip_main_no,
# MAGIC     order_date,
# MAGIC     shop_code,
# MAGIC     customer_tag
# MAGIC   from
# MAGIC     final_sales_table
# MAGIC   where
# MAGIC     order_date >= getArgument("start_date")
# MAGIC     and order_date <= getArgument("end_date")
# MAGIC )
# MAGIC select
# MAGIC   vip_main_no,
# MAGIC   customer_tag,
# MAGIC   count(distinct vip_main_no, order_date, shop_code) as visit
# MAGIC from
# MAGIC   visit0
# MAGIC group by
# MAGIC   vip_main_no,
# MAGIC   customer_tag

# COMMAND ----------

df = spark.sql("""select
  distinct vip_main_no,
  new_joiner_flag,
  visit,
  persona as customer_tag
from
  new_joiner
  inner join visit using (vip_main_no)
  inner join persona using (vip_main_no)
where persona = "Beautyholic"
""")

cluster2_visit = count_pivot_table(df, group_by_col="visit", agg_col="vip_main_no")

# COMMAND ----------

def pivot_table_by_cat(
    table, group_by="item_subcat_desc_cleaned", agg_col="net_amt_hkd", mode="sum"
):
    df = spark.sql(
        f"""
        select * from
            (select 
                distinct 
                case when isnull({group_by}) = 1 or {group_by} = "N/A" then "Unknown" else {group_by} end as {group_by},
                customer_tag, 
                {mode}({agg_col}) as overall_amount
            from {table}
            where order_date >= getArgument("start_date") and order_date <= getArgument("end_date")
            group by 
                customer_tag,
                {group_by}
            )
            PIVOT (
            SUM(overall_amount)
            FOR customer_tag IN ("cluster 1", "cluster 2", "cluster 3", "cluster 4")
            ) 
        """
    )
    display(df)
    return df

# COMMAND ----------

# MAGIC %sql
# MAGIC create
# MAGIC or replace temp view df as
# MAGIC select
# MAGIC   *
# MAGIC from
# MAGIC   final_sales_table
# MAGIC   inner join (
# MAGIC     select
# MAGIC       vip_main_no
# MAGIC     from
# MAGIC       new_joiner
# MAGIC       inner join visit using (vip_main_no)
# MAGIC     where
# MAGIC       new_joiner_flag = 1
# MAGIC       and visit > 1
# MAGIC   ) using (vip_main_no)

# COMMAND ----------

df = spark.table("df")

# COMMAND ----------

count_pivot_table(df, group_by_col="dummy", agg_col="vip_main_no")

# COMMAND ----------

sum_pivot_table(df, group_by_col="dummy", agg_col="net_amt_hkd")

# COMMAND ----------

# 1. amt table by subclass and segment
pivot_table_by_cat("df", group_by="maincat_and_subcat", agg_col="net_amt_hkd", mode="sum")

# COMMAND ----------


