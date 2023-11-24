# Databricks notebook source
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

# COMMAND ----------

# clustering result
persona = spark.read.parquet(
    "/mnt/dev/customer_segmentation/imx/joyce_beauty/model/clustering_result_kmeans_iter1.parquet")
persona.createOrReplaceTempView("persona0")


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
# MAGIC from persona0

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
# MAGIC CREATE OR REPLACE TEMP VIEW sales_cleaned AS
# MAGIC select 
# MAGIC   *,
# MAGIC   case when maincat_desc = "SET" then "Set"
# MAGIC   WHEN maincat_desc = "ZZ" OR maincat_desc = "Dummy" THEN "Unknown" ELSE maincat_desc END AS maincat_desc_cleaned,
# MAGIC   case when item_subcat_desc = "ZZZ" then "Unknown" 
# MAGIC   WHEN item_subcat_desc = "dummy" then "Unknown" 
# MAGIC   WHEN item_subcat_desc = "Dummy" THEN "Unknown"
# MAGIC   WHEN item_subcat_desc = "SET" then "Set" ELSE item_subcat_desc END AS item_subcat_desc_cleaned
# MAGIC from sales

# COMMAND ----------

final_sales_table = spark.sql(
    """
    select *, 1 as dummy, persona as customer_tag from sales_cleaned
    inner join persona using (vip_main_no)
    """
)
final_sales_table.createOrReplaceTempView("final_sales_table")

# COMMAND ----------

# MAGIC %md
# MAGIC demographic

# COMMAND ----------

# count of customer
count_pivot_table(final_sales_table, group_by_col="dummy", agg_col="vip_main_no")

# COMMAND ----------

# gender
df = spark.sql(
    """
    with tem as (select distinct vip_main_no, case when customer_sex = "C" OR isnull(customer_sex) = 1 then "C"
        else customer_sex end as customer_sex_new,
        customer_tag
        from final_sales_table)
        select distinct vip_main_no, min(customer_sex_new) as customer_sex_new, customer_tag from tem group by vip_main_no, customer_tag

    """
)
count_pivot_table(df, group_by_col="customer_sex_new", agg_col="vip_main_no")

# COMMAND ----------

# tenure

# COMMAND ----------

# MAGIC %sql
# MAGIC create
# MAGIC or replace temporary view tenure as
# MAGIC Select
# MAGIC   vip_main_no,
# MAGIC   first_pur_jb,
# MAGIC   round(
# MAGIC     datediff(
# MAGIC       TO_DATE("20231031", "yyyyMMdd"),
# MAGIC       first_pur_jb
# MAGIC     ) / 365,
# MAGIC     0
# MAGIC   ) as tenure
# MAGIC from
# MAGIC   first_purchase

# COMMAND ----------

df = spark.sql("""select
  distinct vip_main_no,
  max(case
    when tenure <= 1 then '0-1'
    when tenure > 1
    and tenure <= 3 then '1-3'
    when tenure > 3
    and tenure <= 7 then '3-7'
    else '8+'
  end) as tenure,
  persona as customer_tag
from
  tenure
  inner join persona using (vip_main_no)
group by 1, 3
""")

count_pivot_table(df, group_by_col="tenure", agg_col="vip_main_no")

# COMMAND ----------

# nationality

df = spark.sql(
    """
    with tem as (select *,
    case when cust_nat_cat = "Hong Kong" then "Hong Kong" 
    when cust_nat_cat = "Mainland China" then "Mainland China" 
    when cust_nat_cat = "Macau" then "Macau" 
    else "Others" end as cust_nat_cat_new
    from final_sales_table)
    select distinct vip_main_no, min(cust_nat_cat_new) cust_nat_cat_new, customer_tag from tem group by vip_main_no, customer_tag
    """
)
count_pivot_table(df, group_by_col="cust_nat_cat_new", agg_col="vip_main_no")

# COMMAND ----------

# shop region
df = spark.sql(
    """
    select distinct vip_main_no, min(region_key) region_key, customer_tag from final_sales_table group by vip_main_no, customer_tag
    """
)
count_pivot_table(df, group_by_col="region_key", agg_col="vip_main_no")

# COMMAND ----------

# age group
df = spark.sql(
    """
    select distinct vip_main_no, max(customer_age_group) customer_age_group, customer_tag from final_sales_table group by vip_main_no, customer_tag
    """
)

table = count_pivot_table(df, group_by_col="customer_age_group", agg_col="vip_main_no").createOrReplaceTempView(
    "age_gp")

# COMMAND ----------

# MAGIC %sql
# MAGIC select 
# MAGIC   distinct
# MAGIC   case 
# MAGIC     when customer_age_group = '01' then '< 25'
# MAGIC     when customer_age_group = '02' then '26 - 30'
# MAGIC     when customer_age_group = '03' then '31 - 35'
# MAGIC     when customer_age_group = '04' then '36 - 40'
# MAGIC     when customer_age_group = '05' then '41 - 50'
# MAGIC     when customer_age_group = '06' then '> 51'
# MAGIC     when customer_age_group = '07' then null
# MAGIC   else null end as age,
# MAGIC   sum(`cluster 1`),
# MAGIC   sum(`cluster 2`),
# MAGIC   sum(`cluster 3`),
# MAGIC   sum(`cluster 4`)
# MAGIC from age_gp
# MAGIC group by age

# COMMAND ----------

# MAGIC %md
# MAGIC transactional

# COMMAND ----------

# amt
df = spark.sql(
    """
    select * from final_sales_table
    where order_date >= getArgument("start_date") and order_date <= getArgument("end_date")
    """
)
sum_pivot_table(df, group_by_col="dummy", agg_col="net_amt_hkd", show_inactive=False)

# COMMAND ----------

# qty
sum_pivot_table(df, group_by_col="dummy", agg_col="sold_qty", show_inactive=False)

# COMMAND ----------

# # of order
count_pivot_table(df, group_by_col="dummy", agg_col="invoice_no", show_inactive=False)

# COMMAND ----------

# # of visit
count_pivot_table(df, group_by_col="dummy", agg_col="invoice_no", show_inactive=False)

# COMMAND ----------

visit = spark.sql(
    """with visit as (
select
  distinct vip_main_no,
  order_date,
  shop_code,
  customer_tag
from final_sales_table
 where order_date >= getArgument("start_date") and order_date <= getArgument("end_date") 
)
select 
  vip_main_no,
  order_date,
  shop_code,
  customer_tag,
  count(distinct vip_main_no,
  order_date,
  shop_code) as visit,
  1 as dummy
from visit
group by
  vip_main_no,
  order_date,
  shop_code,
  customer_tag
""")

sum_pivot_table(visit, group_by_col="dummy", agg_col="visit", show_inactive=False)


# COMMAND ----------

# MAGIC %sql
# MAGIC -- MEMBER PENETRATION BY STORE
# MAGIC -- cust count by store and segment
# MAGIC select * from
# MAGIC (select 
# MAGIC     distinct shop_desc,
# MAGIC     customer_tag, 
# MAGIC     count(distinct vip_main_no) as vip_count
# MAGIC from final_sales_table
# MAGIC where order_date >= getArgument("start_date") and order_date <= getArgument("end_date")
# MAGIC group by 
# MAGIC     customer_tag,
# MAGIC     shop_desc
# MAGIC )
# MAGIC PIVOT (
# MAGIC   SUM(vip_count)
# MAGIC   FOR customer_tag IN ("cluster 1", "cluster 2", "cluster 3", "cluster 4")
# MAGIC ) 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Member penetration by month
# MAGIC -- cust count by yearmon and segment
# MAGIC select
# MAGIC   *
# MAGIC from
# MAGIC   (
# MAGIC     select
# MAGIC       distinct yyyymm,
# MAGIC       customer_tag,
# MAGIC       count(distinct vip_main_no) as vip_count
# MAGIC     from
# MAGIC       (
# MAGIC         select
# MAGIC           *,
# MAGIC           CONCAT(
# MAGIC             year(order_date),
# MAGIC             LPAD(month(order_date), 2, '0')
# MAGIC           ) as yyyymm
# MAGIC         from
# MAGIC           final_sales_table
# MAGIC       )
# MAGIC     group by
# MAGIC       customer_tag,
# MAGIC       yyyymm
# MAGIC   ) PIVOT (
# MAGIC     SUM(vip_count) FOR customer_tag IN ("cluster 1", "cluster 2", "cluster 3", "cluster 4")
# MAGIC   )

# COMMAND ----------

# MAGIC %md
# MAGIC features based on brand, class, subclass

# COMMAND ----------

# - share of wallet 
# - AVERAGE ITEM VALUE
# - MEMBER PENETRATION
# - $ SPEND PER MEMBER

# COMMAND ----------

def pivot_table_by_cat(group_by="item_subcat_desc_cleaned", agg_col="net_amt_hkd", mode="sum",
                       table="final_sales_table"):
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

# by subclass

# COMMAND ----------

# 1. amt table by subclass and segment
pivot_table_by_cat(group_by="item_subcat_desc_cleaned", agg_col="net_amt_hkd", mode="sum")

# COMMAND ----------

# 2. qty table by subclass and segment
pivot_table_by_cat(group_by="item_subcat_desc_cleaned", agg_col="sold_qty", mode="sum")

# COMMAND ----------

# 3. number of member purchase by subclass and segment
pivot_table_by_cat(group_by="item_subcat_desc_cleaned", agg_col="distinct vip_main_no", mode="count")

# COMMAND ----------

# by brand

# COMMAND ----------

# 1. amt table by brand and segment
pivot_table_by_cat(group_by="prod_brand", agg_col="net_amt_hkd", mode="sum")

# COMMAND ----------

# 2. qty table by brand and segment
pivot_table_by_cat(group_by="prod_brand", agg_col="sold_qty", mode="sum")

# COMMAND ----------

# 3. number of member purchase by brand and segment
df = pivot_table_by_cat(group_by="prod_brand", agg_col="distinct vip_main_no", mode="count")

# COMMAND ----------

df.createOrReplaceTempView("brand")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- get brand desc
# MAGIC select b.*, brand_desc from brand b left join imx_prd.imx_dw_train_silver.dbo_viw_lc_xxx_brand_brand on brand_code = prod_brand

# COMMAND ----------

# by maincat_desc

# COMMAND ----------

# 1. amt table by maincat_desc and segment
pivot_table_by_cat(group_by="maincat_desc_cleaned", agg_col="net_amt_hkd", mode="sum")

# COMMAND ----------

# 2. qty table by maincat_desc and segment
pivot_table_by_cat(group_by="maincat_desc_cleaned", agg_col="sold_qty", mode="sum")

# COMMAND ----------

# 3. number of member purchase by maincat_desc and segment
pivot_table_by_cat(group_by="maincat_desc_cleaned", agg_col="distinct vip_main_no", mode="count")

# COMMAND ----------

# MAGIC %md
# MAGIC item tagging

# COMMAND ----------

item_attr = spark.read.parquet(os.path.join(base_dir, "item_attr_tagging.parquet")).toPandas()
item_attr_exploded = item_attr.explode("tags")
item_attr_exploded_spark = spark.createDataFrame(item_attr_exploded)
final_sales_table_with_tags = final_sales_table.join(item_attr_exploded_spark.select("item_desc", "tags"),
                                                     on="item_desc", how="left")

# COMMAND ----------

final_sales_table_with_tags.createOrReplaceTempView("final_sales_table_with_tags")

# COMMAND ----------

# 1. amt table by tag and segment
pivot_table_by_cat(group_by="tags", agg_col="net_amt_hkd", mode="sum", table="final_sales_table_with_tags")

# COMMAND ----------

# 2. qty table by tag and segment
pivot_table_by_cat(group_by="tags", agg_col="sold_qty", mode="sum", table="final_sales_table_with_tags")

# COMMAND ----------

# 3. number of member purchase by tag and segment
pivot_table_by_cat(group_by="tags", agg_col="distinct vip_main_no", mode="count", table="final_sales_table_with_tags")

# COMMAND ----------
