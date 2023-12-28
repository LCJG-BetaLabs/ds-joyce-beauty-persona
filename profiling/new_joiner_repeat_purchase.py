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

cluster_order = ["Beautyholic", "Beauty Accessories and Devices Lover", "Value Shoppers", "Personal Care Enthusiasts"]

# COMMAND ----------


def sum_pivot_table(table, group_by_col, agg_col, show_inactive=True):
    df = table.groupBy("customer_tag", group_by_col).agg(f.sum(agg_col))
    pivot_table = (
        df.groupBy(group_by_col).pivot("customer_tag").agg(f.sum(f"sum({agg_col})"))
    )
    display(pivot_table.select(group_by_col, *cluster_order))
    return pivot_table


def count_pivot_table(table, group_by_col, agg_col, percentage=False, show_inactive=True):
    df = table.groupBy("customer_tag", group_by_col).agg(f.countDistinct(agg_col).alias("count"))
    pivot_table = (
        df.groupBy(group_by_col)
        .pivot("customer_tag")
        .agg(f.sum(f"count"))
    )
    display(pivot_table.select(group_by_col, *cluster_order))
    return pivot_table


# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temp view persona as
# MAGIC select 
# MAGIC   vip_main_no,
# MAGIC   case when persona = 0 then "Beauty Accessories and Devices Lover"
# MAGIC   when persona = 1 then "Beautyholic"
# MAGIC   when persona = 2 then "Value Shoppers"
# MAGIC   when persona = 3 then "Personal Care Enthusiasts" 
# MAGIC   when persona = 4 then "Personal Care Enthusiasts" end as persona -- the 4 and 5 cluster are merged based on profiling result
# MAGIC from persona0

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW sales_cleaned AS
# MAGIC WITH cte1 AS (
# MAGIC select 
# MAGIC   *,
# MAGIC   case when maincat_desc = "SET" then "Set"
# MAGIC   WHEN maincat_desc = "ZZ" OR maincat_desc = "Dummy" THEN "Unknown" ELSE maincat_desc END AS maincat_desc_cleaned,
# MAGIC   case when item_subcat_desc = "ZZZ" then "Unknown" 
# MAGIC   WHEN item_subcat_desc = "dummy" then "Unknown" 
# MAGIC   WHEN item_subcat_desc = "Dummy" THEN "Unknown"
# MAGIC   WHEN item_subcat_desc = "SET" then "Set" ELSE item_subcat_desc END AS item_subcat_desc_cleaned
# MAGIC from sales
# MAGIC ),
# MAGIC Cte2 AS (
# MAGIC   select 
# MAGIC     *,
# MAGIC     concat(maincat_desc_cleaned, " - ", item_subcat_desc_cleaned) as maincat_and_subcat 
# MAGIC   from cte1
# MAGIC )
# MAGIC SELECT * FROM Cte2
# MAGIC WHERE prod_brand not in ("JBGOT", "JB") -- Remove GOTI & Joyce Beauty from the datamart

# COMMAND ----------

final_sales_table = spark.sql(
    """
    select *, 1 as dummy, persona as customer_tag from sales_cleaned
    inner join persona using (vip_main_no)
    """
)
final_sales_table.createOrReplaceTempView("final_sales_table0")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW final_sales_table AS
# MAGIC SELECT
# MAGIC   c.vip_main_no,
# MAGIC   shop_code,
# MAGIC   region_key,
# MAGIC   vip_no,
# MAGIC   invoice_no,
# MAGIC   item_code,
# MAGIC   order_date,
# MAGIC   sold_qty,
# MAGIC   net_amt_hkd,
# MAGIC   net_amt,
# MAGIC   item_list_price,
# MAGIC   shop_brand,
# MAGIC   prod_brand,
# MAGIC   sale_lady_id,
# MAGIC   cashier_id,
# MAGIC   customer_nat,
# MAGIC   cust_nat_cat,
# MAGIC   customer_sex,
# MAGIC   customer_age_group,
# MAGIC   void_flag,
# MAGIC   valid_tx_flag,
# MAGIC   sales_staff_flag,
# MAGIC   sales_main_key,
# MAGIC   sales_type,
# MAGIC   item_desc,
# MAGIC   item_cat,
# MAGIC   item_sub_cat,
# MAGIC   brand_code,
# MAGIC   retail_price_hk,
# MAGIC   retail_price_tw,
# MAGIC   item_product_line_desc,
# MAGIC   maincat_desc,
# MAGIC   item_subcat_desc,
# MAGIC   shop_desc,
# MAGIC   maincat_desc_cleaned,
# MAGIC   item_subcat_desc_cleaned,
# MAGIC   maincat_and_subcat,
# MAGIC   persona,
# MAGIC   dummy,
# MAGIC   CASE
# MAGIC     WHEN c.customer_tag = 'Beauty Accessories and Devices Lover' THEN 'Personal Care Enthusiasts'
# MAGIC     ELSE c.customer_tag
# MAGIC   END AS customer_tag
# MAGIC FROM
# MAGIC   final_sales_table0 c
# MAGIC WHERE
# MAGIC   c.vip_main_no IN (
# MAGIC     SELECT
# MAGIC       p.vip_main_no
# MAGIC     FROM
# MAGIC       final_sales_table0 p
# MAGIC     WHERE
# MAGIC       p.prod_brand = 'JBSLP'
# MAGIC     GROUP BY
# MAGIC       p.vip_main_no
# MAGIC     HAVING
# MAGIC       COUNT(DISTINCT p.prod_brand) = 1
# MAGIC   )
# MAGIC UNION
# MAGIC SELECT
# MAGIC   *
# MAGIC FROM
# MAGIC   final_sales_table0 c
# MAGIC WHERE
# MAGIC   c.vip_main_no NOT IN (
# MAGIC     SELECT
# MAGIC       p.vip_main_no
# MAGIC     FROM
# MAGIC       final_sales_table0 p
# MAGIC     WHERE
# MAGIC       p.prod_brand = 'JBSLP'
# MAGIC     GROUP BY
# MAGIC       p.vip_main_no
# MAGIC     HAVING
# MAGIC       COUNT(DISTINCT p.prod_brand) = 1
# MAGIC   )

# COMMAND ----------

final_sales_table = spark.table("final_sales_table")

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
  persona as customer_tag
from
  new_joiner
  inner join persona using (vip_main_no)
""")

df = df.groupBy("customer_tag", "new_joiner_flag").agg(f.countDistinct("vip_main_no").alias("count"))
pivot_table = (
    df.groupBy("new_joiner_flag")
    .pivot("customer_tag")
    .agg(f.sum(f"count"))
)
display(pivot_table)

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

df = df.groupBy("customer_tag", "visit").agg(f.countDistinct("vip_main_no").alias("count"))
pivot_table = (
    df.groupBy("visit")
    .pivot("customer_tag")
    .agg(f.sum(f"count"))
)
display(pivot_table)

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


