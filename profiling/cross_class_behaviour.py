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

cluster_order = ["Beautyholic", "Beauty Devices Lover", "Skincare Addicts", "Personal Care Enthusiasts"]

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
# MAGIC   case when persona = 0 then "Beauty Devices Lover"
# MAGIC   when persona = 1 then "Beautyholic"
# MAGIC   when persona = 2 then "Skincare Addicts"
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
# MAGIC     WHEN c.customer_tag = 'Beauty Devices Lover' THEN 'Personal Care Enthusiasts'
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
# MAGIC       p.prod_brand in ('JBSLP', 'JBAQU', 'JBRUB')
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
# MAGIC       p.prod_brand in ('JBSLP', 'JBAQU', 'JBRUB')
# MAGIC     GROUP BY
# MAGIC       p.vip_main_no
# MAGIC     HAVING
# MAGIC       COUNT(DISTINCT p.prod_brand) = 1
# MAGIC   )

# COMMAND ----------

final_sales_table = spark.table("final_sales_table")

# COMMAND ----------

import pandas as pd


def get_cross_class_pivot_table(
    customer_tag,
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
        where customer_tag = '{customer_tag}' and
        maincat_desc_cleaned not in ("Life-styled", "Unknown", "Take-In")
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
for i in range(4):
    df = get_cross_class_pivot_table(
        customer_tag=cluster_order[i],
        _class="maincat_desc_cleaned",
        aggfunc=lambda x: len(x.unique()),
        values="vip_main_no",
    )
    display(df)

# COMMAND ----------


